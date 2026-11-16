# PPO self-play baseline — 2026-11-16

## Implementation

Fully trainable 1688→128→128 shared tanh MLP, 2569-action categorical actor, scalar critic. Every mask represents a complete joint action, including preview and replacements. Masked logits are excluded from the distribution; sample/log probability use the exact same Categorical. Waiting players get a dummy distribution and are excluded from actor/entropy terms. The critic trains on player-view states at request rounds. The environment supplies observations for both externally controlled players, with closed sheets and percent opponent HP.

Training uses current-policy self-play on both sides in 16 native vector environments, 32 request rounds per rollout. All observation/mask/reward buffers are copied before the next environment step overwrites them. Terminal reward only (+1/−1/0), gamma=1, GAE lambda=.95, PPO clip=.2, Adam 3e−4, four optimization epochs, minibatches 256, value coefficient .5, entropy coefficient .01, max gradient norm .5, epoch KL early stop .03. There is no distillation, search, auxiliary opponent prediction, reward shaping or historical league. This is a modest fully trainable MLP baseline, not reproduction of the authors' transformers.

For true termination, bootstrap value is zero. For time-limit truncation, bootstrap uses the terminal observation supplied by same-step autoreset. GAE stops across both termination and truncation, preventing returns from leaking into the next game. Nonterminal rollout boundaries bootstrap from the next state. Tests cover these boundaries and exact legal distribution/log probabilities, rewarded toy action/value learning, checkpoint reload and the deployment preview command.

Checkpoints include network dimensions/version, weights, optimizer state, update index, config and RNG metadata. Resume explicitly starts new seeded environments; it does not promise exact mid-run replay. Current code logs loss, approximate KL, gradient norm, completed games, truncations and time. Native action rejections/nonfinite metrics fail the run. Bounds prevent accidental large training runs. Torch and transitive environment versions are pinned in requirements-ppo.lock. All computation used CPU.

Teams are supplied by Pikapro 0.1.0's `sets` generator; no replay data is used. Training and evaluation use separate episode seeds but the same generator distribution. This tests new games, not held-out archetypes. Games from self-play have correlated player trajectories. The initial and final networks face the same evaluation seed distributions; results are not treated as independent paired-outcome proofs or a ladder estimate.

## Measured run

Artifact: `runs/ppo-1794855537055000000/`. Seed 81001, 40 updates, 20,480 batched request rounds / 40,960 player rows. Completed 1,626 games with zero truncations or rejected training actions. Training loop: 6.65 seconds (~3,081 vector-battle request rounds/s, including optimization/checkpoint writes). Startup and evaluation are outside that time. This speed is workload-specific, not the upstream simulator's advertised speedup.

Native evaluation, 100 games per opponent, predeclared seeds 91011/91012:

| Opponent | Initial wins | Trained wins |
|---|---:|---:|
| Random | 45/100 | 83/100 |
| Simple heuristics | 5/100 | 15/100 |

Additional fresh-seed holdout, 200 games per opponent, seeds 111001/111002, initial/final sampling seeds reset equally:

| Opponent | Initial wins | Trained wins |
|---|---:|---:|
| Random | 109/200 | 176/200 |
| Simple heuristics | 1/200 | 24/200 |

No draws/truncations. This demonstrates useful learning relative to initialization in these native matchups; the checkpoint remains weak against the heuristic baseline. No checkpoint selection or tuning used these final results. Holdout details are saved in holdout.json.

## Deployment and limits

`src/agents/ppo_player.py` feeds only the player's protocol history and own request into Pikapro ChannelSession, using the same basic encoding, joint mask and switch-by-row convention as training. Initial local transfer attempt stalled because choice text lacked a Showdown command prefix. The prefix was corrected and covered by a regression test. Server process cleanup was exercised when restarting the stalled run.

The prior two-fixture cross-engine event comparison remains the only deterministic simulator conformance coverage; this work does not establish comprehensive mechanic equivalence. Observation encoding is upstream's `basic`; we have not independently audited every feature. Continue reference validation before expensive runs. Historical checkpoint leagues, richer structured networks, per-archetype held-out pools, additional training seeds and controlled paired cross-engine evaluation remain future work.

Showdown transfer verification: 20 complete games against frozen CFR on balance, seed 101001; 4 wins, zero client warnings/errors. Artifact `runs/1794855639146373000-local-s101001/`. This is an integration smoke test, not a claim of competitive strength.

Approximate marginal 95% Wilson intervals on the 200-game native holdout: random initial 47.6–61.3%, trained 82.8–91.8%; heuristic initial 0.1–2.8%, trained 8.2–17.2%. These describe the observed matchups, without correcting for shared seeds or opponent/team correlations.

Resume smoke test: restored the final model and Adam state, started fresh seed 83001, completed update 41 and 32 complete games without truncations. Artifact `runs/ppo-1794855717281221000/`. This checkpoint was not selected or substituted for the reported 40-update result. A clipped on-policy toy-bandit test also verifies improvement from sampled rewards, alongside the supervised network sanity test.
