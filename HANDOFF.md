# Tyrtlebot handoff — 2026-11-16

Read `VGC_BOT_PLAN.md` and `README.md` first. Goal: a Pokémon Showdown Champions VGC Reg M-C doubles bot, currently closed team sheets and Bo1. Work locally. User prefers self-play reinforcement learning from scratch, like mikumiku/Jaxcalibur; do not replace it with CFR distillation. CFR remains a baseline and possible later inference search.

## Current state

- `src/training/ppo.py`: batched Pikapro PPO self-play, fully trainable MLP policy/value, legal joint masks including preview, terminal rewards, GAE, clipped PPO and checkpoint/optimizer resume. No teacher, search, historical league or human replay training. Pikapro `sets` generates teams. Checkpoint resume starts new seeded games, not exact mid-battle continuation.
- `src/agents/ppo_player.py`: feeds player-visible Showdown protocol into Pikapro ChannelSession and uses the PPO checkpoint. `src/eval/runner.py` runs local games or explicit registered-bot ladder mode. Public login/ladder remains untested; **do not connect publicly without the owner's explicit authorization for the account/run and current bot-policy review**.
- `src/sim/pikapro_backend.py`: native simulator adapter for prior agents; `src/agents/player.py` and `src/search/` implement the original one-turn surrogate MCCFR bot. This search uses heuristic payoffs, not Showdown rollouts or trained full-game CFR. Earlier logistic model and Monte Carlo policy-gradient experiments remain for comparison and are not the main PPO path.
- Pinned Showdown revision: `f827a1d41064e0dec4c19c6ea0394fcf0c23479c`; Pikapro 0.1.0; PyTorch 2.15.0. Setup/versions and commands: `README.md`, `requirements.lock`, `requirements-pikapro.lock`, `requirements-ppo.lock`. Node >=22.18; development used Python 3.14 on macOS arm64.

## Verified results

- Most recent test suite: **22 passed**. PPO 40-update CPU run: 1,626 completed self-play games, zero reported rejected actions/truncations; training loop 6.65 seconds. Saved locally in `runs/ppo-1794855537055000000/`.
- Fresh native holdout, initial → trained: vs random 109/200 → 176/200; vs simple heuristic 1/200 → 24/200. Same generated team distribution, fresh seeds, not held-out archetypes. Learned beyond initialization but remains weak.
- Trained checkpoint completed 20 local Showdown games against frozen CFR: 4 wins, no protocol warnings. PPO checkpoint resume tested. Two seeded Pikapro/Showdown fixtures matched selected public battle events; **comprehensive simulator equivalence and a full observation-leakage audit are not established**.
- See `docs/ppo.md`, `docs/pikapro.md`, `docs/policy.md`, `docs/learning.md`, `docs/validation.md`, `docs/decisions.md` for methods, seeds, limitations and older experiments. All outcomes are local, not evidence of ladder strength.

## Commands

```sh
./scripts/setup.sh
.venv/bin/pip install -r requirements-ppo.lock
.venv/bin/python -m pytest -q
.venv/bin/python -m src.training.ppo --updates 40 --envs 16 --seed 81001
.venv/bin/python scripts/check_native.py
.venv/bin/python -m src.eval.runner local --agent ppo --opponent frozen --model runs/ppo-1794855537055000000/checkpoint.pt --games 20
```

The project is currently **untracked inside an ancestor Git repository** (`git status --short -- .` reports `?? ./`). Avoid touching unrelated ancestor files. `.venv/`, `vendor/`, `runs/`, data and checkpoints are ignored: local checkpoints/results and the simulator checkout will **not** be included in a commit. Reinstall dependencies/recreate the pinned simulator; regenerate models with the training command if those ignored artifacts are unavailable.

## Next work

Audit observation and action encoding/legality and expand differential tests against pinned Showdown; establish diverse team/archetype train/validation/test splits and frozen baselines; evaluate multiple training seeds and meaningful confidence intervals. Add historical-checkpoint self-play, richer trainable policy/value inputs and throughput profiling before considering a larger budget. Preserve pure self-play training and player information boundaries. Do not infer rank-1 readiness from the current smoke tests.
