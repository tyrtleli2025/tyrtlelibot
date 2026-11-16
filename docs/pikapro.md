# Pikapro integration — 2026-11-16

Pikapro 0.1.0 installed from the macOS arm64 abi3 wheel in the project virtual environment. Pin: requirements-pikapro.lock and the optional `pikapro` dependency group in pyproject.toml. License: MIT; upstream includes Pokémon Showdown/poke-env attribution. Source: https://github.com/spedr/pikapro-engine . No upstream code was copied into our source tree.

The native backend maps our exact format to `championsvgc`. It reuses our manually authored fixture. Its player-specific `log('p1'/'p2')` and `request_json(side)` feed separate poke-env DoubleBattle objects. Agents receive only those client views, never native state, teams, seeds or omniscient logs. Each side's choice is computed before either is submitted. Native legal indices filter client-generated actions and validate all submitted choices. Empty ally targeting exposed a difference between client enumeration and native masks; filtering now excludes that choice before selection. Unsupported parser events fail explicitly except for display-only OTS HTML, which is skipped. Closed sheets are not accepted.

The native runner saves manifests, outcomes, both decision traces and player-specific text logs. Its outcome schema is accepted by the existing grouped-split trainer. Native game seeds are base seed plus game index; both agent RNGs are seeded as well. An integration test replays the same seed twice and asserts identical submitted choices and logs. A boundary test changes an unrevealed opposing move while holding the visible team/seed fixed and checks identical player observations. These are focused checks, not a proof covering every mechanic.

Upstream reports conformance against Showdown commit a5df827; our Showdown pin is f827a1d41064e0dec4c19c6ea0394fcf0c23479c. Full cross-engine replay/differential validation is not implemented. Native games are useful for local development, while Showdown remains the reference for deployment. No speed multiplier from the upstream README is assumed to apply to our Python CFR pipeline. Protocol parsing, inference and trace serialization remain in Python.

## Local validation results

15 tests passed. The fixture completed 100 random/random games in 1.79 seconds and 100 CFR/CFR games in 39.48 seconds, with zero agent fallbacks or rejected submissions. This is end-to-end runtime with Python protocol parsing and trace output; differing seeds mean it is not a controlled speedup comparison with earlier Showdown runs.

- `runs/1794851329447891000-local-s30001/`: inference 0.21s of 1.79s elapsed.
- `runs/1794851331590352000-local-s40001/`: inference 37.95s of 39.48s elapsed.

The 20-collection/20-evaluation learning smoke test completed and saved a reloadable model at `runs/experiment-1794851338222614000/outcome.npz`. Its 12/20 evaluation wins are only a pipeline check. No strength improvement is claimed. An initial concurrent-run artifact-discovery failure was fixed by reading the explicit output path from the runner rather than scanning the runs directory.
