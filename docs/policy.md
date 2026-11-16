# Direct policy/value baseline — 2026-11-16

Implementation: `src/training/policy.py`, `scripts/train_policy.py`. A fixed 22×32 random projection plus tanh supplies nonlinear state/action interactions. A 33-parameter actor scores each complete legal joint action; softmax defines the exact sampling distribution. A separate 33-parameter critic predicts terminal returns from state-only features. The two heads train with terminal Monte Carlo returns (win +1, loss −1, draw 0); critic uses squared error. Each game-side trajectory is length-normalized in the update. This is a compact Monte Carlo actor/critic baseline, not PPO or a full trained transformer. The hidden projection is fixed, not trained. No reward shaping, expensive compute or public games.

Both sides use the same frozen batch policy for 20 complete games before a gradient update. Collection has no true opponent sets in policy inputs. The new policy reuses the existing public observation/feature path on native and Showdown backends. It handles forced replacements as masked joint decisions; preview is still random. Legal action selection occurs directly, without CFR search. Critic predictions are used as the learning baseline, not for simulator-rollout search.

Training: 400 games, seed 51001; 20 batches, learning rate .03. The second fixture `teams/native-sample.txt` was generated once using Pikapro 0.1.0 sample_team and frozen as text; it contains Incineroar, Charizard, Garchomp, Gholdengo, Rillaboom and Sinistcha. Both fixture teams pass the pinned Showdown validator. This adds another matchup distribution but is not a held-out team split: both teams are used for training and evaluation. Evaluation uses fresh game seeds, so only game-level generalization is tested.

Training completed in 18.97 seconds. Last-batch value MSE was 0.9579, versus 1.0 initially, and actor norm increased to 0.0503. These changes establish parameter updates, not strength. Artifact: `runs/policy-1794851577707820000/`, including initial.npz, policy.npz, plan.json, metrics.jsonl and report.json.

Focused conformance: `scripts/check_native.py` ran one deterministic battle per fixture under the same Gen5 PRNG seed and choices in native Pikapro and pinned Showdown. Selected spectator switch/move/damage/heal/faint/status/boost/weather/result events matched exactly: 81 events on balance and 75 on native-sample. Artifact: `runs/conformance-1794851641914066000/`. Many mechanics and trajectories remain uncovered. The check preserves reference/native logs and submitted choices for inspection.

Unit tests cover rewarded-action learning, value learning and exact checkpoint reload. All 17 tests pass. The policy remains experimental and is not the default agent.

## Showdown evaluation

Frozen final policy versus frozen original CFR: **1/50 wins** on balance (seed 71001), **3/50 wins** on native-sample (seed 72001). Combined **4/100 wins**, approximate 95% Wilson interval **1.6%–9.8%**. All games completed with zero client protocol warnings/errors. This model is substantially weaker than CFR in this benchmark; it is not promoted. Neither the frozen initial policy nor random policy was tested in this comparison, so it does not measure improvement over initialization.

- `runs/1794851615654704000-local-s71001`: 1.57 ms per policy decision.
- `runs/1794851629767898000-local-s72001`: 1.34 ms per policy decision.

The saved `evaluation.json` records the test paths and results. A larger training run or changed features should be evaluated using fresh seeds, rather than choosing checkpoints on this test. The next useful development is stronger state/action features, more optimization steps verified against an untrained baseline, and then batched-native collection.
