# Satellite Disaster-Damage Triage

[![tests](https://github.com/AhmadRaza4076/DisasterIQ/actions/workflows/test.yml/badge.svg)](https://github.com/AhmadRaza4076/DisasterIQ/actions/workflows/test.yml)

Team **DarkNem** — AMD Developer Hackathon ACT II (Track 3: Unicorn)

Automated building damage assessment from pre/post disaster satellite imagery. Deterministic ML scoring is the source of truth; Fireworks LLM narrates ranked zone reports only.

## Architecture

- **frontend/** — Next.js UI (upload, canvas overlay, situation brief)
- **backend/** — FastAPI (`/analyze`, `/brief`, `/health`)
- **ml/** — xView2 baseline inference (Docker)
- **data/** — xBD test set + curated demo pairs

## Prerequisites

| Tool | Install | Verify |
|------|---------|--------|
| **Node.js 18+** | https://nodejs.org/ or `winget install OpenJS.NodeJS.LTS` | `node --version` |
| **Python 3.12** | `winget install Python.Python.3.12` | `%LOCALAPPDATA%\Programs\Python\Python312\python.exe --version` |
| **WSL 2** | Run **as Administrator**: `.\scripts\install-wsl-admin.ps1` then **reboot** | `wsl --status` |
| **Docker Desktop** | https://www.docker.com/products/docker-desktop/ (after WSL) | `docker ps` |

Check everything: `.\scripts\verify-prerequisites.ps1`

**Note:** Disable Windows Store Python aliases if `python` fails: Settings → Apps → Advanced app settings → App execution aliases → turn off `python.exe` and `python3.exe`.

## Quick start (local dev)

```powershell
# Backend (Windows venv at backend/.venv/Scripts/python.exe)
.\scripts\start-backend.ps1

# Frontend (separate terminal)
.\scripts\start-frontend.ps1
```

Or manually:

```powershell
cd backend
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000

cd frontend
npm install
npm run dev
```

Open http://localhost:3000

**Environment file:** Copy `.env.example` to `.env` at the **repo root** (gitignored). The backend loads it automatically via `backend/app/config.py`.

**Demo day (recommended):** Run the backend with `.\scripts\start-backend.ps1` (local venv), not `docker compose up`, especially when `INFERENCE_MODE=docker`. The containerized backend image does not include the Docker CLI, so ML inference from inside Compose will fail.

Rehearse before recording:

```powershell
.\scripts\rehearse-demo.ps1
```

## Docker (optional full stack)

Requires Docker Desktop installed.

```powershell
cp .env.example .env
# Set FIREWORKS_API_KEY when hackathon credits unlock

# Build ML inference image (optional, ~15-30 min first time)
docker compose --profile build-ml build ml

docker compose up --build
```

> **Note:** `INFERENCE_MODE=docker` requires the Docker CLI on the host. Use `start-backend.ps1` for real ML inference during demos; use Compose for frontend + stub backend smoke tests only.

- Frontend: http://localhost:3000
- Backend: http://localhost:8000
- API docs: http://localhost:8000/docs

## Environment variables

| Variable | Description |
|----------|-------------|
| `FIREWORKS_API_KEY` | Fireworks AI API key (optional; stub used if unset) |
| `FIREWORKS_MODEL` | Model id (default: `accounts/fireworks/models/glm-5p2`) |
| `DEMO_DATA_DIR` | Path to demo image pairs |
| `INFERENCE_MODE` | `stub` (default) or `docker` |
| `XVIEW2_DOCKER_IMAGE` | Baseline inference image name |

## ML baseline weights

Pretrained weights download automatically when building the inference Docker image. **TF 1.15 on ROCm is not supported** — see [ml/README.md](ml/README.md) for the framework decision before AMD GPU fine-tuning.

```powershell
docker compose --profile build-ml build ml
```

Dockerfile: `ml/inference/Dockerfile` (patched TF 1.15 base; clones upstream baseline inside the image).

Release: https://github.com/DIUx-xView/xview2-baseline/releases/tag/v1.0

## Demo data

Curated pairs in `data/demo/` (earthquake + flood from xBD test set). To recreate on another machine:

```powershell
.\scripts\curate_demo_subset.ps1
# or from tar only:
.\scripts\curate_demo_subset.ps1 -TarPath D:\path\to\test_images_labels_targets.tar
```

See `data/demo/manifest.json` for the exact 10 pair IDs.

## Team collaboration

- Friend setup: [docs/FRIEND_SETUP.md](docs/FRIEND_SETUP.md)
- Work split: [docs/TEAM_ROLES.md](docs/TEAM_ROLES.md)
- Disk / Docker on D: [docs/DISK_SPACE.md](docs/DISK_SPACE.md)
- Data layout: [docs/DATA.md](docs/DATA.md)
- Submission checklist: [docs/SUBMISSION.md](docs/SUBMISSION.md)

## License

MIT (application code). xView2 baseline under BSD-3 (see `ml/xview2-baseline/LICENSE.md`).
