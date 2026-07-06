# Hackathon submission checklist — DisasterIQ

**Repo:** https://github.com/AhmadRaza4076/DisasterIQ  
**Team:** DarkNem · **Track:** AMD ACT II Unicorn

## Before submitting on lablab.ai

- [ ] Public GitHub repo is up to date
- [ ] README has setup steps and architecture summary
- [ ] Demo video (2–3 min) recorded
- [ ] `.env` secrets are **not** in git (`FIREWORKS_API_KEY` local only)

## Demo video script (outline)

1. **Problem** (20s) — Pakistan 2022 floods / earthquakes; need fast zone triage from satellite imagery
2. **Upload** (30s) — Select earthquake or flood demo pair from xBD
3. **Analyze** (45s) — Show damage overlay + ranked zone table (destroyed / major counts)
4. **Brief** (30s) — Situation brief for coordinators (Fireworks or stub)
5. **Architecture** (30s) — ML scores deterministically; LLM narrates only, never re-ranks
6. **Closing** (15s) — GitHub link, team name

## Fireworks API key

When hackathon credits unlock:

```powershell
Copy-Item .env.example .env   # if needed
# Edit .env — set FIREWORKS_API_KEY=your_key_here
# Restart backend
```

Stub briefs work without the key for recording; live narration is nicer for judges.

## Optional stretch goals

- `INFERENCE_MODE=docker` after ML image builds
- Fine-tuned weights on AMD GPU (Phase 3)
- Friend walkthrough per [FRIEND_SETUP.md](FRIEND_SETUP.md)
