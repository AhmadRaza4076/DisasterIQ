# Hackathon submission checklist — DisasterIQ

**Repo:** https://github.com/AhmadRaza4076/DisasterIQ  
**Team:** DarkNem · **Track:** AMD ACT II Unicorn

## Before submitting on lablab.ai

- [x] Public GitHub repo is up to date
- [x] README has setup steps and architecture summary
- [ ] Demo video (2–3 min) recorded
- [x] `.env` secrets are **not** in git (`FIREWORKS_API_KEY` local only)
- [ ] lablab.ai submission with GitHub URL + video link

## Rehearse locally

```powershell
# Terminal 1
.\scripts\start-backend.ps1

# Terminal 2
.\scripts\start-frontend.ps1

# Terminal 3 — smoke test API + brief
.\scripts\rehearse-demo.ps1
```

Use `INFERENCE_MODE=stub` in `.env` for fast, reliable masks during recording.

## Demo video script (outline)

1. **Problem** (20s) — Pakistan 2022 floods / earthquakes; need fast zone triage from satellite imagery
2. **Upload** (30s) — Select earthquake or flood demo pair from xBD
3. **Analyze** (45s) — Show damage overlay + ranked zone table (destroyed / major counts)
4. **Brief** (30s) — Situation brief for coordinators (Fireworks live or `fireworks-fallback` if API hiccups)
5. **Architecture** (30s) — ML scores deterministically; LLM narrates only, never re-ranks
6. **Closing** (15s) — GitHub link, team name

## Fireworks API key

1. Copy `.env.example` → `.env` at repo root (never commit `.env`)
2. Paste your **full** Fireworks API key from the hackathon credits page (not the placeholder in `.env.example`)
3. Restart backend after editing `.env`

If the brief badge shows `fireworks-fallback` or `stub`, the key may be invalid or the model unavailable — stub brief still works for recording.

## Optional stretch goals

- `INFERENCE_MODE=docker` via `start-backend.ps1` (not `docker compose up` backend)
- PyTorch fine-tune on AMD GPU per [AMD_FINETUNE_PLAN.md](AMD_FINETUNE_PLAN.md)
- Friend walkthrough per [FRIEND_SETUP.md](FRIEND_SETUP.md)
