# Decisions — 2026-11-16

## Scope

The initial directory contained only VGC_BOT_PLAN.md and .DS_Store. Git resolves to an ancestor repository; no parent files or Git state were changed. No applicable ancestor AGENTS.md was found. This implementation is a first functional slice, not completion of the rank-1 plan.

Use `gen9championsvgc2026regmc`: Pokémon Champions, VGC 2026 Regulation M-C, doubles, bring six/select four, best-of-one. The pinned server's format exists and uses the `champions` mod. Closed sheets: the client declines optional OTS. Forced OTS and Bo3 are rejected by configuration rather than silently treated as closed/Bo1. Current public availability still requires checking at deployment.

## Dependencies and fixtures

Pokémon Showdown MIT, revision `f827a1d41064e0dec4c19c6ea0394fcf0c23479c`, version 0.11.11; Node >=22.18.0 required, tested 22.19.0 with npm 10.9.3. Its committed package-lock.json is used through npm ci. poke-env 0.17.0 is MIT; exact Python transitive versions are recorded in requirements.lock. Tested on macOS arm64, Python 3.14.2. No GPU, paid compute, private data or pretrained checkpoints.

`teams/balance.txt` is a manually authored smoke-test fixture, not a competitive recommendation or imported team dataset. It uses Champions stat points (66 total, max 32/stat). Initial SV-style EVs, Amoonguss and Incineroar's Knock Off failed the reference validator and were corrected. The final team passes TeamValidator for the exact format. There is only one team: no held-out strength evaluation or generalization claim.

## Algorithm

External-sampling Monte Carlo CFR on a simultaneous one-turn zero-sum **surrogate** game. Each iteration samples each player's action from regret matching, evaluates every counterfactual action against the sampled opponent, updates both players' cumulative regrets, and accumulates average strategies. Both policies adapt during each solve; this is local matrix-game self-play. Play samples the average strategy. No neural model, persistent full-game regret table, or trained checkpoint is implied.

Candidate rows are legal complete joint orders from poke-env's request parser. Keep top immediate damage/utility candidates plus exploratory candidates. Columns are hypothetical joint opponent attack/Protect intentions formed from visible types, revealed moves and a stated generic 80-power STAB prior. Sample damage variation and speed tie breaks, then solve the resulting averaged payoff matrix. Unknown opponent sets are never read from the local server or opposing client. The solver is independently checked on matching pennies and a dominated-action game.

This intentionally departs from the plan's later learned-value/reference-simulator search milestone to deliver the user's requested modest CFR prototype now. **It is not reference-simulator rollout search, a calibrated hidden-set posterior, or full extensive-form Pokémon MCCFR.** Surrogate mechanics must not be mistaken for simulator correctness. Authoritative matches still run in Showdown.

The surrogate models type effectiveness, STAB, spread scaling, target selection, HP depletion, focus-fire knockouts, priority/base-speed ordering, Tailwind, a repeated-Protect success approximation, and Fake Out first-turn eligibility. Damage is a heuristic, not a damage calculator. It does not faithfully model abilities, items, exact stats, boosts, field/weather/status, redirection, switch-in transitions, secondary effects, mega stat changes, or opponent switches. These are known search limitations. Team preview and forced switches currently use seeded random legal selection. Standard-gen-9 poke-env data supplies type/base-stat features; Champions-specific changes are authoritative only on the server. Do not claim Milestone 1 mechanics coverage or Milestone 6 search completion.

## Operations

Local runner owns a loopback-only server process group and stops all workers on success/failure. Local guest authentication and throttling exceptions exist only in that isolated server config. Agent seeds are deterministic; ordinary WebSocket challenges do not expose a simulator RNG seed, so battles are **not bit-for-bit reproducible**. Manifests explicitly record simulator_seed=null. Results include replay logs, per-decision player views, chosen orders, time, warnings and source/team hashes. Runs are ignored by Git. Secrets are environment-only and omitted from manifests.

Public play is an explicit separate `ladder` command, with registration and run-authorization acknowledgements and credentials. The implementation does not register accounts, conceal automation, or change opt-in restrictions. No public connection was made during development. Public account/authentication/queue behavior remains untested.

## Policy checked 2026-11-16

https://www.smogon.com/forums/threads/initial-policy-on-bots.3789528/ says registration is mandatory; October 12 update provides `/ht` → Something else → I want to register my bot. October 20 update says humans opt in to bots and bot matches do not affect human ratings. Recheck at deployment; historical rank-1 examples are not equivalent to the current pool.
https://pokemonshowdown.com/rules prohibits gaming the rating system. Only local self-play is used here.

## Sources and licenses

- https://github.com/smogon/pokemon-showdown — MIT, source and npm lock pinned locally; retain its LICENSE on redistribution.
- https://poke-env.readthedocs.io/en/stable/ — MIT, player/client/action API; installed source inspected to verify doubles and preview behavior.
- https://github.com/smogon/pokemon-showdown/blob/master/PROTOCOL.md — WebSocket battle protocol, handled by poke-env.
- Plan's mikumiku37/Nessie/Jaxcalibur reports are motivation only. No author code, data or numerical performance claim is used.

Milestones 2–7 remain future research work; this prototype does not establish strength, exploitability of actual Pokémon, or ladder rank.

## Local learning extension — 2026-11-16

Added a bounded persistent-learning experiment authorized by the owner. NumPy logistic regression avoids another heavyweight dependency. It learns action-conditioned outcome correlations from actual local matches, with equal per-battle weight, battle-grouped validation, and a separate fresh evaluation seed. A small fixed score blends into CFR. The original decision logic is preserved in `src/agents/frozen.py`; default `cfr` remains available. Observation schema v2 adds legal known-roster/status/field information; no hidden simulator reconstruction is performed. This is a first supervised outcome baseline, not completion of the original PPO/value-rollout milestone. The script predeclares evaluation before collection, saves a model artifact and reports a Wilson interval instead of assuming improvement.

## Pure self-play PPO extension

Following the owner's clarification, the new learning path trains from scratch using only terminal game outcomes in batched current-policy self-play. It does not distill CFR. Fully trainable PyTorch MLP and joint categorical masks replace the prior fixed-feature policy for this mode. CPU-first defaults are deliberately bounded. Past-checkpoint league and auxiliary opponent heads are deferred; policy search remains a separate later inference feature. The earlier experimental models and default CFR remain available for comparisons. Methods, dependencies, native holdout counts and Showdown transfer results are in docs/ppo.md.
