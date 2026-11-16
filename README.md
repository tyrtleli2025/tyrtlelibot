# Tyrtlebot

A functional CPU VGC bot for Pokémon Showdown, using external-sampling Monte Carlo CFR self-play in a small one-turn tactical game. Actual battles run on the pinned Showdown simulator. The search uses approximate damage/utility, not full simulator rollouts or a trained value network. See [decisions and limitations](docs/decisions.md).

Target: **Champions VGC 2026 Reg M-C**, closed team sheets, best-of-one. This is distinct from Scarlet/Violet. Includes a manually authored, reference-validated six-Pokémon team, legal doubles actions, preview, forced replacements, Mega Evolution, local matches, replays, decision traces and a public ladder adapter.

## macOS setup

Use Python 3.14 and Node >=22.18 (tested Python 3.14.2, Node 22.19.0). From this directory:

```sh
./scripts/setup.sh
```

The script installs the exact Python lock, checks out the recorded Showdown revision, runs npm ci/build, validates the fixture, and runs unit tests. The underlying commands were exercised during development. A local virtual environment and simulator checkout are already present in this workspace.

## Local tests and games

```sh
.venv/bin/python -m pytest -q
node scripts/validate-team.cjs
.venv/bin/python -m src.eval.runner local --games 20
.venv/bin/python -m src.eval.runner local --games 20 --opponent cfr --seed 42
.venv/bin/python -m src.eval.runner local --games 20 --agent random
```

The runner starts its own server at `127.0.0.1:8000`, completes the batch, and stops its workers. An occupied port is an error; adjust `port` in `configs/local.json`. If an execution sandbox blocks local sockets, run these locally with socket permissions. Do not run two local batches on the same port.

Each `runs/<timestamp>-local-s<seed>/` contains `manifest.json`, `results.json`, `decisions.jsonl`, `server.log` and HTML replays. Failed runs also write `failure.json`. Warnings/invalid choices make a batch fail rather than count as a clean pass. Agent randomness is seeded; server battle RNG is not controlled, so repeated runs can differ.

Search width and CFR iterations are configurable. The agent keeps a mixture of damage-ranked and exploratory legal joint candidates, samples tactical uncertainty, and samples from the average regret-matching policy. CFR restarts at each decision; `--opponent cfr` additionally runs full games against the same algorithm. There is no expensive training step or checkpoint to download.

## Registered public ladder test

No public run was made. First register your bot through Showdown `/ht` → Something else → I want to register my bot, verify the format/account with staff and recheck [current policy](https://www.smogon.com/forums/threads/initial-policy-on-bots.3789528/). Current policy uses human opt-in and says bot games do not affect human ratings. The pool can be small; this is not a historical unrestricted rank-1 attempt.

Set your registered account credentials in your shell without saving them in the repo, then explicitly launch a bounded run:

```sh
export PS_USERNAME='your-registered-bot-name'
read -s 'PS_PASSWORD?Showdown password: '
export PS_PASSWORD
.venv/bin/python -m src.eval.runner ladder --games 1 --registered-bot --authorize-public-run
unset PS_PASSWORD
```

The password prompt syntax above is for macOS zsh. Authentication and WebSocket transport use poke-env's public Showdown configuration; the local server's guest/throttle settings are never used remotely. A 900-second overall timeout also bounds waiting for an opponent; change it in config if needed. Public mode is implemented but unverified without an authorized registered account. If a live run times out/disconnects midgame, inspect the account/battle before retrying; automatic recovery is not implemented.

## Limits

This is a starting bot, not an optimal policy. Team preview and forced switches are random legal choices. The tactical model omits many VGC interactions and has no persistent learning. Only closed-sheet Bo1 is supported; other modes fail explicitly. One fixture team is not a representative benchmark. The tests establish specific solver and integration behavior, not complete Pokémon mechanics coverage. See [validation results](docs/validation.md) for exactly what was run.

## Small local learning experiment

```sh
.venv/bin/python scripts/experiment.py
```

This runs a bounded CPU experiment: 200 CFR self-play collection games, a battle-grouped 80/20 training/validation split, then 100 fresh games against the frozen original bot. The model, data paths, source hashes, seeds, plan and evaluation report are saved under `runs/experiment-*/`. The evaluation seed is separate, but ordinary WebSocket battles still have unseeded simulator randomness. Repeating the command repeats the procedure, not exact battle trajectories. Do not run concurrent batches on the same port.

The 22-feature logistic model predicts eventual wins from the player's visible position and chosen joint action. Each battle has equal training weight; mirrored player views of a game stay in the same split. The observation now includes known rosters (including fainted Pokémon), public statuses/boosts, weather, fields and side conditions, with unknown moves/items remaining unknown. The current model uses a subset of those fields. Opponent preview roster entries do not imply that those Pokémon were brought to battle.

At inference, `learned` mode adds a small predicted-outcome preference to each candidate row of the existing CFR payoff matrix. This is an action-conditioned outcome predictor under the collection policies; it is not a simulator rollout value function or a causal estimate of how much changing an action helps. Collection still uses actual Showdown outcomes. Preview and forced replacements remain random.

Use the saved model directly:

```sh
.venv/bin/python -m src.eval.runner local --agent learned --opponent frozen --model runs/experiment-<timestamp>/outcome.npz --games 100 --seed 92001
```

The benchmark predeclares a 95% Wilson lower bound above 50% against the frozen bot as its improvement criterion. It uses one fixture team, so it cannot establish generalization across teams. A failed criterion is reported as no demonstrated improvement; the default `cfr` agent is unchanged. See [learning results](docs/learning.md).

## Pikapro native training backend

Install the pinned optional simulator:

```sh
.venv/bin/pip install -r requirements-pikapro.lock
.venv/bin/python -m src.eval.runner local --backend pikapro --games 20
.venv/bin/python -m src.eval.runner local --backend pikapro --opponent cfr --games 100
.venv/bin/python scripts/experiment.py --backend pikapro
```

This uses Pikapro 0.1.0 directly in-process, with no local server or sockets. The existing CFR/random/frozen/learned agents run through a player-specific protocol adapter. Both decisions are collected before submission. Native legality is checked before selecting and submitting actions. `--agent learned --model <path>` works with either backend. The collection/training/evaluation pipeline writes the same artifact schema, so native games can train our existing model.

Native manifests record the backend, package version and simulator seed (base seed plus game index). Results contain submitted choices and seeds; `replays/` contains each player's text protocol log instead of HTML. Native battles use Pikapro's seeded fast RNG. They are repeatable within the pinned build, but do not use Showdown's default RNG. The default backend remains Showdown, and public ladder mode requires Showdown. The native adapter currently supports only our configured Champions Reg M-C closed-sheet Bo1 format and fixture.

Pikapro is MIT licensed: [upstream source, license and notices](https://github.com/spedr/pikapro-engine). Its authors report conformance against a different Showdown revision from ours. We have verified legal complete games, repeatability and player-view isolation locally, **not full differential equivalence** to our pinned Showdown. See [native backend validation](docs/pikapro.md).

## Direct policy/value self-play

```sh
.venv/bin/python scripts/train_policy.py --games 400
.venv/bin/python scripts/check_native.py
.venv/bin/python -m src.eval.runner local --agent policy --opponent frozen --model runs/policy-<timestamp>/policy.npz --games 50
```

The first command trains a small nonlinear policy/value model on CPU using terminal Monte Carlo policy gradients with a learned baseline. Both players use the same fixed policy during each 20-game collection batch, then update together. It uses two fixed teams, chosen before evaluation. This is a simpler first baseline than PPO. It has a fixed random tanh feature layer and trained actor/value heads, uses the existing observable action features, and samples a single categorical distribution over complete legal joint actions. Preview remains random; forced switches are handled by the policy. No Python CFR solve runs in policy mode.

Initial/final checkpoints, training curves, seeds and team/source hashes are saved under `runs/policy-*/`. `--resume <policy.npz>` resumes the learned parameters; use a new `--seed` to collect new games. It does not resume an interrupted game or RNG stream. Collection runs battles sequentially with native simulation; native vector-environment integration remains future work.

The conformance command runs two seeded native/reference battle fixtures and compares selected mechanics-bearing spectator events. It is a focused compatibility check, not exhaustive simulator validation. See [policy experiment results](docs/policy.md).

## Batched PPO self-play from scratch

```sh
.venv/bin/pip install -r requirements-ppo.lock
.venv/bin/python -m src.training.ppo --updates 40 --envs 16 --seed 81001
```

This is the current pure self-play learning path: a fully trainable PyTorch policy/value network, Pikapro's native batched player-visible encoder, complete joint action masks (including preview), terminal rewards, PPO clipping, GAE, entropy regularization, gradient clipping and a KL stop. No CFR teacher, human replays or search are used in training. Both sides use the current policy; a historical league is not yet implemented. The environment generates varied legal teams using Pikapro's `sets` source.

Checkpoints include model/optimizer state and schema dimensions. Resume on a fresh environment seed:

```sh
.venv/bin/python -m src.training.ppo --resume runs/ppo-<timestamp>/checkpoint.pt --updates 40 --seed 82001
```

Resume continues parameters and optimizer, not an interrupted battle/RNG stream. The bounded CLI accepts up to 200 updates and 64 environments per run. Default CPU training uses two Torch threads and two simulator threads. No GPU or public connection is started.

A checkpoint can play through the existing Showdown runner using Pikapro's ChannelSession encoder:

```sh
.venv/bin/python -m src.eval.runner local --agent ppo --opponent frozen --model runs/ppo-<timestamp>/checkpoint.pt --games 20
```

PPO deployment currently uses the Showdown backend; native PPO evaluation is performed directly by the batched trainer. The older `policy` and `learned` modes use different checkpoint formats. See [PPO methods and measured results](docs/ppo.md).
