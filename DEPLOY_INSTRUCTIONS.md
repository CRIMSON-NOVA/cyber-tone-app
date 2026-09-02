# Deploying CYBER TONE

This app has two parts that deploy separately:

- **`backend/`** — FastAPI + WebSocket + the two PyTorch detection models.
  This needs to run as an always-on process, so it goes on **Render** or
  **Railway** (not Vercel — see note at the end for why).
- **`frontend/`** — the dashboard (`index.html`). This is plain static
  HTML/JS, so it deploys perfectly on **Vercel**.

Deploy the backend first, since the frontend needs its URL.

---

## Step 1: Push this folder to GitHub

Open this whole `cyber-tone-deploy` folder in your IDE, open its built-in
terminal, and run:

```bash
git init
git add .
git commit -m "Initial CYBER TONE deploy"
git branch -M main
git remote add origin https://github.com/<your-username>/cyber-tone.git
git push -u origin main
```

(Create the empty `cyber-tone` repo on GitHub first if you haven't — same as
before, just skip adding any files through the web UI this time; the push
above does that for you, folder structure and all.)

---

## Step 2: Deploy the backend (Render)

1. Go to https://render.com → **New +** → **Web Service**
2. Connect your GitHub repo (`cyber-tone`)
3. Set:
   - **Root Directory:** `backend`
   - **Environment:** `Docker` (it will detect the `Dockerfile` automatically)
   - **Instance Type:** free/starter tier is fine to test
4. Add environment variables:
   - `FRONTEND_ORIGIN` = `https://your-app.vercel.app` (once you know it — see Step 3)
   - `ACCESS_TOKEN` = a long random private string only you know (this locks
     the WebSocket down so only someone with this token can connect — see
     the Security section below)
5. Click **Create Web Service**

First deploy takes a few minutes (installing torch/transformers, downloading
both models on first boot). Once live, you'll get a URL like:

```
https://cyber-tone-backend.onrender.com
```

Your WebSocket endpoint is that URL with `wss://` and the path:

```
wss://cyber-tone-backend.onrender.com/ws/detect
```

Test it's alive by visiting `https://cyber-tone-backend.onrender.com/` in a
browser — you should get a small JSON status response.

---

## Step 3: Deploy the frontend (Vercel)

1. Go to https://vercel.com → **Add New** → **Project**
2. Import the same GitHub repo
3. Set **Root Directory** to `frontend`
4. Framework preset: **Other** (static HTML, no build step)
5. Click **Deploy**

You'll get a URL like `https://cyber-tone.vercel.app`.

---

## Step 4: Connect them

Open your deployed Vercel URL. Paste in your Render WebSocket URL from
Step 2, plus your access token as a query parameter, e.g.:

```
wss://cyber-tone-backend.onrender.com/ws/detect?token=YOUR_ACCESS_TOKEN
```

Click **Save**, then **Start Microphone**. It's saved in your browser, so you
only need to do this once per device.

---

## Notes

- **Cold starts:** Render's free tier spins the backend down after inactivity.
  The first connection after idle time can take 30–60+ seconds while it wakes
  up and reloads both models.
- **CORS/Origin:** `FRONTEND_ORIGIN` restricts which site can connect over
  HTTP. Set it to your real Vercel URL once you have it.
- **HTTPS/WSS:** Vercel serves over HTTPS, so the browser will only allow
  `wss://` connections from it — make sure your backend URL uses `wss://`.

### Why not put everything on Vercel?

Vercel's WebSocket support runs inside serverless functions: each connection
is pinned to one instance for a limited max duration, there's no guarantee of
hitting the same instance twice, and there's no GPU. That's a bad fit here
because two ~300MB+ PyTorch models need to stay loaded in memory across a
long-lived streaming session. Render/Railway run this as a normal always-on
container instead, which is what this workload needs.
