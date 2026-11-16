# Rank-1 VGC Bot: Project Plan for Codex

Last updated: 2026-11-16

## Purpose and instructions for Codex

Build a reproducible Pokémon Showdown VGC doubles agent with the long-term goal of reaching rank 1 in a specifically chosen, authorized ladder. Make measurable progress in small, testable increments. Strength, simulator correctness, and evaluation quality all matter; there is no guarantee of a rank-1 result.

Treat this as a brand-new, greenfield project. No private bot code, simulator, models, datasets, checkpoints, or infrastructure are available. Build the implementation from scratch using publicly available, appropriately licensed dependencies and data. The owner is Shao, author of mikumiku37; use that project as a research reference, not a code dependency. Begin with Milestone 0 and the first vertical slice. Inspect the working directory and any `AGENTS.md` before changing files, and preserve any unrelated user files you find.

Work locally by default. Do not launch expensive training, acquire paid compute, access private datasets, publish source/weights, or connect a bot to a public server without the owner's authorization. Record decisions, commands, relevant dependency versions, seed values, and results. Do not commit secrets, credentials, model checkpoints, or large datasets unless explicitly directed.

### First response and first work session

1. Inspect the working directory, git status if applicable, and any `AGENTS.md`. Initialize the project structure below without overwriting unrelated files. Record the selected format and assumptions in `docs/decisions.md`.
2. Set up a Python package and reproducible dependencies; pin a compatible Pokémon Showdown revision and Python packages after verifying current installation requirements. Add configuration, ignore generated artifacts, and document setup for macOS.
3. Create a minimal local-only end-to-end path: a configurable local Showdown server; legal doubles teams from a documented source or small manually authored fixtures; a random legal-action agent; and a command to run a small batch of battles and save results. Use a verified format identifier supported by the pinned server. Exercise team preview, joint action selection, forced switches, and battle completion. If dependencies or permissions block execution, document the exact blocker and commands.
4. Add a short README with environment setup, exact commands actually verified, how to run local tests, and known limitations. Report what was run and the next milestone.

## Goal, constraints, and definitions

**Competitive target:** rank 1 on one named Pokémon Showdown VGC ladder under the rules in force at the time of evaluation. Report format, regulation, platform/game, best-of-one or best-of-three, team-sheet visibility, dates, rating, rank, games played, account eligibility, and policy conditions. Avoid conflating Scarlet/Violet VGC with Pokémon Champions; their mechanics and format identifiers may differ.

**Default development target:** Champions VGC 2026 Reg M-C, best-of-one, closed team sheet, on a pinned local Showdown revision if still available. This is a research starting point derived from mikumiku37, not a claim it is an active or eligible public ladder. Confirm the actual target with the owner before committing to expensive format-specific work or a public run. Make format, team visibility, and series length explicit configuration, not implicit constants. If the target uses open team sheets or best-of-three, expose only the information actually available and track game-to-game state as required.

**External policy:** Check current Showdown/Smogon rules and bot registration, disclosure, matchmaking and permission requirements before any public account or queue. Respect opt-in restrictions, rate limits and competitive integrity rules. Do not impersonate humans, conceal a bot, manipulate ratings, use unauthorized assistance in tournaments, or bypass restrictions. A registered bot's pool may differ from the historical ladder population; report that and any scarcity of opponents. If rank 1 is unavailable under current policy, pursue local benchmarks and consensual challenges and clearly report the external blocker.

**Hardware:** Begin on the available MacBook (CPU first; use Apple GPU only if the stack supports it and profiling justifies it). Support a remote Linux/NVIDIA worker as an optional later path. Do not assume 48 hours on any laptop reproduces a 5090 run. Benchmark simulator throughput, inference, utilization, memory, wall-clock and cost before scaling.

## Research anchors and how to use them

These are author-reported examples, not guarantees or exact specifications. Check linked sources and current format rules before relying on them. The project starts from scratch; do not ask for or depend on an existing mikumiku37 implementation.

- [mikumiku37: rank-1 VGC bot](https://www.smogon.com/forums/threads/mikumiku37-a-self-play-vgc-bot-reached-1-on-the-vgc-reg-m-c-ladder.3789199/): an approximately 8.7M-parameter transformer trained with PPO and a league of previous checkpoints, without human replay training; one-turn inference search over candidate joint actions and sampled hidden worlds. The author reported approximately 330M games over 48.3 hours on a 5090 with a custom simulator. Exact replication requires details beyond the forum report; code and weights were not offered as public dependencies.
- [Nessie123: rank-1 OTS VGC bot](https://www.smogon.com/forums/threads/nessie123-an-ots-vgc-bot-that-topped-the-reg-m-c-bo3-ladder.3789213/): reported a ~1.5M-parameter evaluator, ~375k self-play games, ~20 hours on an M4 Mac mini plus $120 cloud compute, using search-guided learning. Its training assumptions and OTS/Bo3 format differ from closed-sheet Bo1; do not compare hardware or game counts as controlled experiments.
- [Jaxcalibur: rank-1 random-battles bot](https://www.smogon.com/forums/threads/jaxcalibur-a-gen-9-randbats-bot-that-reached-1-on-the-ladder.3787537/): an example of self-play policy/value learning plus search under hidden information, but singles random battles are not VGC.
- [Pokémon Showdown source and simulator](https://github.com/smogon/pokemon-showdown), [protocol](https://github.com/smogon/pokemon-showdown/blob/master/PROTOCOL.md), and [poke-env documentation](https://poke-env.readthedocs.io/en/stable/): integration and reference behavior. Pin revisions; verify actual APIs and license obligations.
- [Showdown rules](https://pokemonshowdown.com/rules) and [initial bot policy](https://www.smogon.com/forums/threads/initial-policy-on-bots.3789528/): recheck before public deployment. Policy can change.
- [PPO paper](https://arxiv.org/abs/1707.06347), [Gymnasium documentation](https://gymnasium.farama.org/), [PyTorch tutorials](https://pytorch.org/tutorials/), and [Stable-Baselines3](https://stable-baselines3.readthedocs.io/): useful background and starting tools, not requirements to replicate another implementation.

## Architecture

Keep these boundaries explicit, with typed Python interfaces and versioned schemas:

1. **Simulator adapter:** reset seeded battle, obtain each player's legal observation and actions, submit simultaneous joint choices, advance to the next decision, save outcomes, and optionally clone or restore state for search. Showdown is the reference. A faster engine is optional and must pass differential validation.
2. **Information boundary:** internal full state remains inside the simulator. Construct a separate observation for each player containing only information that player can legally know, including own team, public preview/reveals, field state, battle history, and allowed team-sheet data. Never feed true unrevealed opponent sets, bench choices, or internal random seeds to policy, value, or search. Explicit unknown masks distinguish unseen from absent.
3. **Legal action generator:** produce complete legal joint actions, including moves and valid targets, switches, mechanics allowed by the selected ruleset, team preview and forced replacements. Apply joint constraints (e.g., avoid selecting the same bench Pokémon twice or using a once-per-battle mechanic twice). Version a stable action encoding and legal mask; assert every chosen action is legal. Handle one active slot, empty target slots, disabled/locked moves and simultaneous replacement.
4. **Agent:** `act(observation, legal_actions, rng)`; implementations for random, heuristics, trained policy and search. For training, expose joint-action log probabilities, value prediction and legality masks using the same rules as inference.
5. **Experiment runner:** separate processes/configs for collection, training, evaluation, and public integration; artifact manifests record source revision, simulator/format revision, team dataset split, model version, and seeds. No public connection by default.

Initial repository structure: `src/{sim,observations,actions,agents,training,search,eval}/`, `configs/`, `tests/`, `scripts/`, `docs/`, and ignored `data/`, `runs/`, `checkpoints/`.

## Milestones

### 0. Scope, infrastructure, and benchmark baseline

- [ ] Choose a target ruleset, visibility and evaluation mode; identify suitable public dependencies and team sources, review their licenses and usage terms, and pin exact code and data versions.
- [ ] Start a local Showdown server and run two scripted agents. Prefer `poke-env` as an adapter if its current doubles/preview API supports the chosen format; do not assume it provides fast vectorized training or perfectly matches a custom simulator.
- [ ] Implement deterministic seed/config handling where supported, structured per-game logs, timeouts and clean process shutdown.
- [ ] Track games/hour, simulator and inference time, CPU/GPU load, memory, errors, and cost. Measure startup versus steady-state performance.

**Exit:** 20 consecutive local complete games with legal teams, no illegal action or crash, clear battle logs and a repeatable command. This is an integration gate, not a strength claim.

### 1. Observation, team preview, and action correctness

- [ ] Define a serializable player-view schema, legal joint actions, masks and terminal results. Cover bring/select rules, lead order, forced switches and doubles targeting for the actual target format.
- [ ] Create targeted reference fixtures for priority, speed ties, Protect, spread/redirected moves, status, field effects, abilities, items and ruleset-specific mechanics. Test limits and switching constraints.
- [ ] Test legal action encoding/decoding and prevent hidden-information leakage. Generate observations after controlled reveals and compare two full states that should look identical to the agent.
- [ ] Add random and simple heuristic bots, such as damage preference plus basic Protect and switching. Keep strength claims separate from validity.

**Exit:** fixture suite passes against pinned Showdown; zero illegal decisions in at least 100 local games. Document unimplemented mechanics and ensure unsupported mechanics fail explicitly rather than silently approximate.

### 2. Evaluation before optimization

- [ ] Set aside team pools by team or archetype and distinct seed sets for training, validation and held-out tests; avoid related variants appearing across splits. Record provenance, collection dates and any usage terms.
- [ ] Create frozen random, heuristic and historical-checkpoint opponents; evaluate both sides, multiple teams and seeds. Save per-opponent win/draw/loss, win-rate confidence intervals, game duration, crashes, and illegal actions. For comparisons use paired matchup/seed assignments where feasible; do not treat turns or games sharing a series as independent.
- [ ] Start with 100–200 games per comparison as a noisy smoke test; increase according to a predeclared confidence interval and minimum meaningful difference. As a rough guide, a 50% win rate over 200 independent games has about ±7 percentage points 95% uncertainty; correlations reduce effective sample size. Report counts and intervals rather than claiming small gains are proven.
- [ ] Never use the test set to tune weights or pick checkpoints repeatedly; reserve an additional final holdout or refresh it if tuning leaks into it.

**Exit:** one command reproduces a versioned benchmark report and detects a deliberately broken legality or information-boundary check.

### 3. Train a small policy/value baseline

- [ ] Establish a minimal PyTorch policy/value network over structured features, initially small enough for local iteration. Define masked probabilities over complete joint actions or a carefully conditioned factorization; confirm sampled actions and log probabilities correspond exactly to the same joint distribution. Preview is a separate decision type or explicitly encoded action space.
- [ ] Implement a small PPO or supported masked-PPO prototype. Validate the algorithm first on a toy environment and then on simple fixed opponents. Correctly handle simultaneous decisions, legal masks, trajectory boundaries, bootstrapping and terminal outcomes. Start with win/loss/draw terminal reward; evaluate any shaping separately for unintended incentives.
- [ ] Log loss, entropy, KL, learning rate, gradient norms, throughput, reward distribution, checkpoint metrics and random seeds. Save optimizer state for resumability. Smoke-test loading and deterministic inference.
- [ ] Compare against random and heuristic opponents plus an untrained policy. Train multiple seeds where feasible and report baseline results, compute, and wall time.

**Exit:** a checkpoint demonstrably learns in the toy test and completes legal VGC games, with a repeatable evaluation against frozen baselines. Training stability and actual win rates must be reported; no promised threshold.

### 4. League self-play and generalization

- [ ] Add opponent sampling from current, recent and diverse historical checkpoints, plus scripted opponents; freeze checkpoints and their configs. Track the distribution and age of opponents to reduce cycling and forgetting.
- [ ] Sample legal teams from a documented, diverse pool, using only permitted sources. Keep held-out teams and archetypes outside training and opponent selection. Train preview selection jointly or as a clearly evaluated separate policy.
- [ ] Evaluate a cross-play matrix, including old checkpoints and held-out opponents. Report exploitability indicators and per-archetype weaknesses; do not equate beating the latest self-play opponent with overall improvement.
- [ ] If native Showdown becomes a bottleneck, benchmark alternatives first. Keep the reference backend for validation; do not promise a fixed throughput multiplier.

**Exit:** measurable, reproducible improvement over the frozen previous model across a predeclared evaluation suite, with uncertainty and regressions disclosed.

### 5. Fast simulator, only if profiling warrants it

- [ ] Prototype the smallest supported subset using a suitable existing engine if licensed and compatible, or a custom engine. Correctness is a feature: pin reference behavior and preserve full state needed by effects and history.
- [ ] Build differential tests: initialize identical known states and actions; compare legal choices and observable outcomes with Showdown. For deterministic cases compare exact transitions; for randomness use aligned RNG only if possible and otherwise compare empirical distributions with predeclared tolerances. Save minimal reproductions for mismatches.
- [ ] Cover actual data distribution, multi-turn interactions, rare mechanics and malformed/unsupported states. Run end-to-end cross-engine agent matches. Track mismatches by mechanic, and fall back to the reference or exclude unsupported rules rather than silently train on wrong physics.
- [ ] Benchmark end-to-end training throughput including serialization, data transfer and inference, not simulator steps in isolation. Decide CPU/GPU/cloud based on the measured bottleneck.

**Exit:** documented coverage and differential-test results, explicit known limitations, and a measured end-to-end throughput gain before large-scale training.

### 6. Search and hidden information

- [ ] Establish the no-search policy/value baseline at fixed checkpoints, teams and compute budgets. Add one-turn search behind a feature flag only after clone/restore or an equivalent reliable simulator interface exists.
- [ ] Generate a diverse top-k set of **legal complete joint actions** for each side, including defensive and switching options. For hidden opponent actions, ensure candidate generation uses that player's legal information, not secrets available only to the simulator.
- [ ] Maintain an explicit posterior or transparent prior over hidden sets, leads and reserves consistent with preview, team-sheet rules and revealed history. Sample plausible worlds, enforce consistency, and document approximation. Do not access true hidden game state during inference; do not leak information across hypothetical worlds.
- [ ] For each sampled world and action pair, simulate one simultaneous turn, average over random outcomes, evaluate resulting states with the learned value function, and aggregate an action payoff matrix. Clearly define value perspective and any discounting. Use an appropriate zero-sum mixed-strategy solver or a tested approximation; validate it on known small matrix games. Sample actions from the resulting distribution when intended.
- [ ] Control candidate width, number of worlds and RNG samples, batching, decision deadline and deterministic fallbacks. Compare policy-only, search-only variants and search budgets on the same held-out evaluation pool. Add deeper search or opponent modeling only if ablations justify the cost and avoid strategy fusion/information leakage.

**Exit:** search never emits illegal actions or uses unavailable information, meets the latency budget, and shows a measured gain over policy-only with an honest confidence interval and reported compute cost.

### 7. Scale and authorized competitive evaluation

- [ ] Choose a training run only after reviewing learning curves, simulator fidelity, evaluation variance and cost estimates. Fix a budget, checkpoint cadence and stop conditions with the owner. Estimate full training and inference cost separately.
- [ ] Freeze the candidate, team-selection policy and configuration before final evaluation. Use held-out local matchups and consenting human challengers when available. Audit failures, disconnected games, legality, latency and rule compliance.
- [ ] Before public laddering, check current registration, disclosure, matchmaking, allowed formats and operator requirements. Obtain the owner's explicit go-ahead for the specific account and run. Never automate around a policy or queue restriction.
- [ ] Record publicly verifiable peak rank with timestamp, exact format, rating, W/L, matchup pool, code/model hashes and any policy limitation. If rank 1 cannot be fairly or legally measured, report the best available authorized benchmark and the reason.

## Failure modes to investigate continuously

- **Simulator mismatch:** high simulated win rate, poor Showdown results; diff trajectories and mechanics before adding model capacity.
- **Information leakage:** policy exploits opponent private team or RNG; test observational equivalence and audit all feature pipelines and search inputs.
- **Invalid doubles actions:** joint switches, targeting, preview and format-specific once-per-game rules are common pitfalls; test the exact encoded joint distribution.
- **Self-play collapse:** strong against latest clone but weak against old checkpoints or simple strategies; use league cross-play and diverse teams.
- **Evaluation noise:** avoid attributing small rating or win-rate differences to a change without enough independent games and matched conditions.
- **Search-model mismatch:** simulated worlds or values differ from real observations; compare calibrated predicted outcomes with recorded results and test search ablations.
- **Format drift:** a new regulation, game or policy invalidates assumptions; pin versions and revalidate before claiming performance.

## Definition of done

A rank-1 claim requires documented evidence of the named authorized ladder, rank and timestamp plus a reproducible model/config and honest record of games and restrictions. Engineering completion also requires legal action handling, zero known hidden-state leakage, documented simulator fidelity, a repeatable benchmark, operational safeguards, and a concise reproduction guide. If external policy or resources prevent a rank-1 attempt, explicitly separate that limitation from completed engineering milestones.
