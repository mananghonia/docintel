# Deploying DocIntel

The app is a single self-contained Docker image (see [`Dockerfile`](Dockerfile)):
Django serves the React app + API in one process, predicts in-process, and runs
tasks eagerly — no Redis/worker/model-server. A champion model is baked in at
build time, so extraction works on the first request.

Two paths below. **Fly.io** is the recommended one — it deploys the `Dockerfile`
directly and has a free-friendly allowance.

---

## Option A — Fly.io (recommended, ~5 minutes)

Everything except creating your Fly account is copy-paste. A config file
[`fly.toml`](fly.toml) is already in the repo.

### 1. Install the CLI

```bash
# macOS / Linux
curl -L https://fly.io/install.sh | sh
```
```powershell
# Windows (PowerShell)
iwr https://fly.io/install.ps1 -useb | iex
```

### 2. Sign up / log in (this is the step only you can do)

```bash
fly auth signup      # or: fly auth login
```

Fly asks for a card to prevent abuse, but the small shared-cpu-1x/1GB machine
this uses fits inside their free allowance and auto-sleeps when idle.

### 3. Pick a unique app name

Edit the first line of `fly.toml`:

```toml
app = "docintel-<yourname>"     # must be globally unique
primary_region = "bom"          # optional: a region near you
```

### 4. Create the app (without deploying yet)

```bash
fly apps create docintel-<yourname>
```

### 5. Set the secret key

```bash
# generate one and store it as a Fly secret (never in the repo)
fly secrets set DJANGO_SECRET_KEY="$(python -c 'import secrets; print(secrets.token_urlsafe(50))')"
```

### 6. Deploy

```bash
fly deploy
```

Fly builds the image (this runs the model-bake step, ~1–2 min), pushes it, and
starts the machine. When it finishes:

```bash
fly open        # opens https://docintel-<yourname>.fly.dev
```

That's it — the dashboard loads with seeded demo docs, and "Ingest 10 synthetic"
or a real invoice upload works immediately.

### Useful follow-ups

```bash
fly logs                         # tail logs
fly deploy                       # redeploy after a git change
fly scale memory 2048            # bump RAM if you enable heavier tiers
```

To **persist data across redeploys** (verified docs, promoted models), add a
volume — otherwise each redeploy starts fresh from the baked model (fine for a
demo):

```bash
fly volumes create docintel_data --size 1 --region bom
```

then add to `fly.toml`:

```toml
[mounts]
  source = "docintel_data"
  destination = "/app/data"
```

### Lock it down (optional)

The demo is open by design. To require a login token:

```bash
fly secrets set REQUIRE_AUTH=1
fly deploy
fly ssh console -C "python backend/manage.py create_api_token --username you --password secret"
# then in the browser console:
#   localStorage.setItem("docintel_token", "<token>"); location.reload()
```

---

## Option B — any Docker host / VPS

```bash
docker build -t docintel .
docker run -d -p 8000:8000 -e DJANGO_SECRET_KEY="$(openssl rand -base64 40)" docintel
# with persistence:
docker compose -f docker-compose.deploy.yml up --build -d
```

Then put it behind your reverse proxy (Caddy/nginx) for TLS. Same image works on
Render's Docker runtime, Google Cloud Run, Railway, etc. — point the host at the
`Dockerfile` and expose port 8000.

---

**Note:** the image and configs are verified by running every build step locally
(prod serving, `collectstatic`, model bake, WSGI import) but have **not been
built inside Docker on the author's machine** (no Docker daemon there). The first
real `docker build` / `fly deploy` may surface an environment detail — if it
does, the fix is usually a one-line Dockerfile change.
