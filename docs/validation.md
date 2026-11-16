# Validation — 2026-11-16

## Verified

- `node scripts/validate-team.cjs`: fixture legal in pinned Champions Reg M-C.
- `.venv/bin/python -m pytest -q`: **9 passed**.
- `.venv/bin/pip install -e .`: package and `tyrtlebot --help` work.
- `.venv/bin/python -m compileall -q src`: passed.
- Public command without explicit registration/run acknowledgements exits before any connection.
- Local batches below: every game completed, zero client warnings/errors, no illegal-action reports. Both sides have saved replays. Preview, forced switches, one-active-slot decisions, and Mega Evolution were exercised.

| Agents | Agent seed | Games | P1 wins | Wall seconds including server lifecycle | Mean / max decision ms (P1) | Forced switches (P1) |
|---|---:|---:|---:|---:|---:|---:|
| cfr vs random | 20261116 | 20 | 20 | 6.27 | 17.07 / 39.46 | 21 |
| cfr vs cfr | 42 | 100 | 45 | 40.90 | 15.57 / 49.31 | 181 |
| cfr vs random | 20261117 | 20 | 20 | 5.76 | 17.51 / 41.58 | 19 |
| random vs random | 20261118 | 20 | 12 | 2.58 | 0.16 / 0.66 | 31 |

The 100-game self-play batch recorded 62 P1 Mega Evolution choices. This is an integration test, not a complete mechanics test suite. 45/100 P1 wins in self-play is not evidence that either identical algorithm is stronger. The 20/20 wins against random use the same fixture on both sides and are not a held-out competitive evaluation. No ranking, rating or optimality claim.

The final 20-game CFR/random and 20-game random/random batches include the completed fallback/error-reporting and process-resource logging changes. The 100-game run used the same tactical policy before those operational additions. Exact code and team hashes live in each manifest. Runtime is on this Mac only, with no paid compute. Timings include startup; they are not isolated simulator-throughput measurements. Python CPU and peak RSS are measured in final results; server CPU/memory utilization and GPU profiling are not implemented.

## Artifacts

- `runs/1794849227918663000-local-s20261116/`
- `runs/1794849333423421000-local-s42/`
- `runs/1794849392266459000-local-s20261117/`
- `runs/1794849398376691000-local-s20261118/`

`results.json`, `manifest.json`, `decisions.jsonl`, `server.log`, and `replays/` are kept locally and ignored by Git. The manifest seeds only agent randomness; simulator seeds are explicitly unknown. Results cannot be reproduced bit-for-bit through ordinary challenge requests.

## Failures found and resolved

1. Network sandbox blocked dependency downloads; dependencies were installed with approved network access.
2. Local socket/worker binding initially failed with EPERM; validation used approved loopback server access.
3. An initial multi-game attempt hit Showdown's ten-second challenge cooldown and was interrupted. The isolated local server now sets `nothrottle`; this never applies to public connections.
4. Initial SV-style team was invalid under Champions. Converted to legal stat points/species/moves and verified with TeamValidator.

The simulator's pinned npm dependency install reported 13 audit findings (1 low, 5 moderate, 7 high). Versions were not automatically upgraded away from the verified pin. The development server binds only to loopback; it is not an internet-hosting deployment.

## Not verified / next milestone

Public account login and ladder queue require a registered account and authorized run. Full mechanics fixtures, held-out teams, confidence-interval evaluation, simulator-seeded repeatability, full-state reconstruction from legal observations, reference-engine search rollouts, learned values and persistent training remain future work. The next useful improvement is a better observation-conditioned tactical model, tested on a separate team pool against frozen baselines.
