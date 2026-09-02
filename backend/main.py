import asyncio
import gc
import json
import logging
import os
import time
import numpy as np
import torch
from collections import deque
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

# Limit PyTorch CPU threads to prevent memory fragmentation on low-RAM containers
torch.set_num_threads(1)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("CyberTone-Backend")

app = FastAPI(title="CYBER TONE - Real-Time Voice Cloning Defense Engine")

# Allow your Vercel frontend (and local dev) to connect. Set FRONTEND_ORIGIN
# as an env var on your host (e.g. https://your-app.vercel.app) once deployed.
# "*" works for testing but is not recommended for production.
FRONTEND_ORIGIN = os.environ.get("FRONTEND_ORIGIN", "*")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_ORIGIN] if FRONTEND_ORIGIN != "*" else ["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Private access key: set this in Render's environment variables (never in
# code / never committed to GitHub). If unset, the backend is open to
# anyone with the URL — fine for a first test, not recommended beyond that.
ACCESS_TOKEN = os.environ.get("ACCESS_TOKEN")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 1. LOAD OPTIMIZED DETECTOR MODEL
# ==========================================
# MelodyMachine V2 is the top performing deepfake audio classifier (~99.7% accuracy).
# Quantized to 8-bit (qint8) to run smoothly inside 512 MB RAM environments.
DETECTOR_MODEL_NAMES = [
    "MelodyMachine/Deepfake-audio-detection-V2",
]

ENSEMBLE_STRATEGY = "max"


def load_detector(model_name: str):
    """Load audio-classification model + feature extractor with 8-bit quantization
    to run smoothly within lightweight memory footprints."""
    logger.info(f"Loading {model_name} on {DEVICE} (optimized memory mode)...")
    extractor = AutoFeatureExtractor.from_pretrained(model_name)
    mdl = AutoModelForAudioClassification.from_pretrained(
        model_name,
        low_cpu_mem_usage=True
    ).to(DEVICE)
    mdl.eval()

    # Apply 8-bit dynamic quantization to linear layers (reduces RAM from ~380MB to ~95MB)
    if DEVICE.type == "cpu":
        try:
            mdl = torch.quantization.quantize_dynamic(mdl, {torch.nn.Linear}, dtype=torch.qint8)
            logger.info("Applied 8-bit dynamic quantization successfully.")
        except Exception as q_err:
            logger.warning(f"Quantization skipped: {q_err}")

    gc.collect()

    id2label = mdl.config.id2label
    idx_of_fake = None
    for idx, label in id2label.items():
        if any(term in str(label).lower() for term in ["fake", "spoof", "synthetic", "ai", "clone", "generated"]):
            idx_of_fake = int(idx)
            break

    if idx_of_fake is None:
        raise ValueError(
            f"Could not identify a 'fake' class in {model_name}'s labels: {id2label}. "
            "Update the matching terms or set the index manually."
        )

    logger.info(f"[{model_name}] using label '{id2label[idx_of_fake]}' (index {idx_of_fake}) as the spoof class")
    return extractor, mdl, idx_of_fake


DETECTORS = [load_detector(name) for name in DETECTOR_MODEL_NAMES]
gc.collect()


# Only windows at/above this RMS energy are treated as "actively speaking."
# Anything below it (silence, room tone) is treated as paused and is never
# run through the model. Kept LOW deliberately so whispers still get
# analyzed — the trailing-tail spike problem is handled separately below
# (crest-factor gating + outlier-robust smoothing), not by hiding quiet
# audio behind a high threshold.
SILENCE_RMS_THRESHOLD = 0.005

# Crest factor = peak amplitude / RMS energy over a window. Sustained speech
# (even whispers) has a moderate crest factor. A sharp transient — a knock,
# a mic bump, a tap — has a very short high peak against low average energy,
# so its crest factor spikes much higher. Windows above this are treated as
# an impulsive noise event and skipped entirely rather than fed to the model.
IMPULSE_CREST_FACTOR_THRESHOLD = 12.0

# If a single window's raw spoof probability jumps more than this much from
# the current smoothed value, it's clamped rather than trusted outright.
# Prevents one spurious frame (echo, brief noise, model hiccup) from
# yanking the displayed/frozen reading around.
OUTLIER_CLAMP_DELTA = 0.35

# ==========================================
# 2. AUDIO BUFFER
# ==========================================
class StabilizedAudioBuffer:
    def __init__(self, sample_rate=16000, window_duration=2.0, step_duration=0.5):
        self.sample_rate = sample_rate
        self.window_size = int(sample_rate * window_duration)
        self.step_size = int(sample_rate * step_duration)
        self.buffer = np.array([], dtype=np.float32)

    def append(self, pcm_bytes: bytes):
        chunk = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        self.buffer = np.append(self.buffer, chunk)

    def has_next_window(self) -> bool:
        return len(self.buffer) >= self.window_size

    def get_window(self) -> np.ndarray:
        window = self.buffer[:self.window_size]
        self.buffer = self.buffer[self.step_size:]
        return window


# ==========================================
# 3. INFERENCE (run off the event loop)
# ==========================================
def run_single_detector(extractor, mdl, target_index, norm_audio: np.ndarray) -> float:
    """Blocking model call for one detector. Always invoke via
    asyncio.to_thread so it doesn't stall the event loop / other concurrent
    WebSocket connections."""
    inputs = extractor(norm_audio, sampling_rate=16000, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    with torch.no_grad():
        logits = mdl(**inputs).logits
        probs = torch.softmax(logits, dim=-1)
    return float(probs[0][target_index].item())


async def run_ensemble(norm_audio: np.ndarray) -> tuple[float, dict]:
    """Runs every detector in DETECTORS concurrently and combines their
    scores per ENSEMBLE_STRATEGY. Returns (combined_prob, per_model_scores)."""
    tasks = [
        asyncio.to_thread(run_single_detector, extractor, mdl, idx, norm_audio)
        for extractor, mdl, idx in DETECTORS
    ]
    scores = await asyncio.gather(*tasks)
    per_model = {
        DETECTOR_MODEL_NAMES[i].split("/")[-1]: round(scores[i], 4)
        for i in range(len(scores))
    }

    if ENSEMBLE_STRATEGY == "max":
        combined = max(scores)
    else:
        combined = float(np.mean(scores))

    return combined, per_model


# ==========================================
# 4. WEBSOCKET ENDPOINT
# ==========================================
@app.websocket("/ws/detect")
async def websocket_detection_endpoint(websocket: WebSocket):
    # Private access gate: if ACCESS_TOKEN is set on the server, only
    # connections that supply the matching ?token=... query param are
    # accepted. Anyone without it is rejected before the connection is even
    # opened. Browsers can't send custom headers on a native WebSocket, so
    # the token travels as a query param — treat it like a password: don't
    # share the full wss://...?token=... URL publicly.
    if ACCESS_TOKEN:
        provided_token = websocket.query_params.get("token")
        if provided_token != ACCESS_TOKEN:
            logger.warning("Rejected WebSocket connection: missing/incorrect access token")
            await websocket.close(code=4401)
            return
    else:
        logger.warning(
            "ACCESS_TOKEN is not set — this backend is OPEN to anyone with the URL. "
            "Set ACCESS_TOKEN as an environment variable to restrict access."
        )

    await websocket.accept()
    stream_buffer = StabilizedAudioBuffer(sample_rate=16000, window_duration=2.0, step_duration=0.5)
    score_window = deque(maxlen=3)

    # Holds the most recent *active* (non-silence) reading so the meter can
    # freeze on it during pauses instead of resetting to 0 / reacting to
    # background noise. Starts at a neutral "nothing heard yet" state.
    last_active_response = {
        "spoof_probability": 0.0,
        "status": "STANDBY",
        "recommended_action": "LISTENING",
        "model_scores": {},
    }

    try:
        while True:
            data = await websocket.receive_bytes()
            stream_buffer.append(data)

            while stream_buffer.has_next_window():
                audio_window = stream_buffer.get_window()

                rms_energy = float(np.sqrt(np.mean(audio_window ** 2)))

                # Below SILENCE_RMS_THRESHOLD: treat as paused, skip inference entirely.
                if rms_energy < SILENCE_RMS_THRESHOLD:
                    # Do NOT clear score_window here: keeping the smoothing
                    # history means that when speech resumes, the running
                    # average isn't restarted from a single noisy sample.
                    response = {
                        "spoof_probability": last_active_response["spoof_probability"],
                        "rms_energy": round(rms_energy, 4),
                        "status": "PAUSED" if last_active_response["status"] != "STANDBY" else "STANDBY",
                        "recommended_action": last_active_response["recommended_action"],
                        "timestamp_ms": int(time.time() * 1000),
                        "frozen": True,
                        "silence_threshold": SILENCE_RMS_THRESHOLD,
                        "model_scores": last_active_response["model_scores"],
                    }
                else:
                    peak = float(np.max(np.abs(audio_window)))
                    crest_factor = (peak / rms_energy) if rms_energy > 1e-6 else 0.0

                    if crest_factor > IMPULSE_CREST_FACTOR_THRESHOLD:
                        # Sharp transient (knock, mic bump, tap) riding on top
                        # of otherwise normal audio. Not sustained speech —
                        # skip inference and hold the frozen reading instead
                        # of letting a bang get classified as a voice.
                        response = {
                            "spoof_probability": last_active_response["spoof_probability"],
                            "rms_energy": round(rms_energy, 4),
                            "status": "NOISE_IGNORED",
                            "recommended_action": last_active_response["recommended_action"],
                            "timestamp_ms": int(time.time() * 1000),
                            "frozen": True,
                            "silence_threshold": SILENCE_RMS_THRESHOLD,
                            "model_scores": last_active_response["model_scores"],
                        }
                        await websocket.send_text(json.dumps(response))
                        continue

                    norm_audio = audio_window / peak if peak > 0 else audio_window

                    try:
                        raw_prob, per_model_scores = await run_ensemble(norm_audio)
                    except Exception as inference_error:
                        logger.error(f"Inference failed: {inference_error}")
                        # Skip this window rather than crashing the whole connection
                        continue

                    # Outlier clamp: if this frame disagrees wildly with the
                    # recent smoothed value, cap how far it can move things
                    # in one step rather than trusting it outright.
                    if len(score_window) > 0:
                        current_median = float(np.median(score_window))
                        deviation = raw_prob - current_median
                        if abs(deviation) > OUTLIER_CLAMP_DELTA:
                            raw_prob = current_median + np.clip(
                                deviation, -OUTLIER_CLAMP_DELTA, OUTLIER_CLAMP_DELTA
                            )

                    score_window.append(raw_prob)
                    # Median instead of mean: a single bad frame can't drag
                    # the displayed value around the way an average can.
                    running_prob = float(np.median(score_window))

                    if running_prob >= 0.70:
                        status = "CLONED_VOICE_DETECTED"
                        action = "ALERT_USER"
                    elif running_prob >= 0.40:
                        status = "SUSPICIOUS"
                        action = "CHALLENGE_REQUIRED"
                    else:
                        status = "AUTHENTIC_HUMAN"
                        action = "ALLOW"

                    response = {
                        "spoof_probability": round(running_prob, 4),
                        "rms_energy": round(rms_energy, 4),
                        "status": status,
                        "recommended_action": action,
                        "timestamp_ms": int(time.time() * 1000),
                        "frozen": False,
                        "silence_threshold": SILENCE_RMS_THRESHOLD,
                        "model_scores": per_model_scores,
                    }
                    last_active_response = {
                        "spoof_probability": response["spoof_probability"],
                        "status": response["status"],
                        "recommended_action": response["recommended_action"],
                        "model_scores": per_model_scores,
                    }

                await websocket.send_text(json.dumps(response))

    except WebSocketDisconnect:
        logger.info("Client disconnected")
    except Exception as e:
        logger.error(f"Stream error: {str(e)}")
        try:
            await websocket.close()
        except RuntimeError:
            # Socket may already be closed/closing
            pass


# ==========================================
# 5. HEALTH CHECK & FRONTEND STATIC SERVING
# ==========================================
from fastapi.responses import FileResponse
import pathlib

FRONTEND_INDEX = pathlib.Path(__file__).resolve().parent.parent / "frontend" / "index.html"


@app.get("/")
async def root_endpoint():
    if FRONTEND_INDEX.exists():
        return FileResponse(FRONTEND_INDEX)
    return {
        "status": "ok",
        "service": "CYBER TONE backend",
        "device": str(DEVICE),
        "detectors": DETECTOR_MODEL_NAMES,
        "websocket_endpoint": "/ws/detect",
    }


@app.get("/health")
@app.get("/healthz")
async def health_check():
    return {
        "status": "ok",
        "service": "CYBER TONE backend",
        "device": str(DEVICE),
        "detectors": DETECTOR_MODEL_NAMES,
        "websocket_endpoint": "/ws/detect",
    }



if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

