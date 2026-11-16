# Local learning experiment

The prototype now has persistent learning through a 22-parameter NumPy logistic outcome model (plus its bias within those parameters). Input is an action-conditioned feature vector containing active/known-team HP and counts, active statuses, Tailwind, turn, and candidate attack/Protect/switch/priority/utility summaries. The broader version-2 observation also serializes boosts, known rosters, weather, fields and public side conditions for future feature work. Private requests, unknown sets, server seeds and actual opponent selections are not inputs.

The model is fitted from real Showdown outcomes. Each game contributes equal total training weight. Both sides' observations of a game are grouped into the same training or validation partition. There is no turn-random split, and benchmark test data is rejected by the trainer. Unknown opponent bench selection means known-team summaries are imperfect; the model cannot treat every preview Pokémon as a confirmed reserve.

The learned score adds `0.25 * (2 * predicted_win_probability - 1)` to each row of the CFR matrix. It influences choices among candidates retained by the current search; it does not replace the legal action generator or improve preview. It is intentionally a limited first experiment and may fail to improve battle strength despite learning outcome correlations.

`src/agents/frozen.py` preserves the original bot decision logic. It shares the unchanged tactical model and legal generator, while ignoring the new model. The added observation fields do not alter its tactical scoring. Manifests record hashes of the entire source tree, fixture and model.

Experiment command: `.venv/bin/python scripts/experiment.py`. Default budget: 200 collection games, 600 inexpensive full-batch gradient steps, 100 evaluation games. Agent seeds: collection 11001, evaluation 91001, training split 123. Match outcomes depend on unseeded server RNG. No paid compute or public connections.

Evaluation is predeclared in plan.json before any test game. Candidate is tested against the frozen baseline on the same fixture team, with a fresh seed. A 95% Wilson lower confidence bound above 0.5 is required to report demonstrated improvement. No repeated checkpoint selection uses test results. This is an initial one-team smoke benchmark; multi-team, swapped-side paired seeded matches and a final holdout remain future work.

## First measured result

Executed the default experiment successfully: **200 collection games and 100 test games**, all complete with zero client protocol warnings/errors. The candidate won **54/100** against the frozen bot; its 95% Wilson interval is **44.3%–63.4%**. **Improvement was not demonstrated** under the predeclared criterion. The model is available as an experimental option; the default bot was not promoted to it.

Training: 160 games / 3,949 player positions. Validation: 40 games / 955 player positions. Equal-per-game log loss: train 0.5844, validation 0.5571, versus 0.6931 for a constant 50% predictor. Better outcome prediction does not imply better action choice: observations and chosen actions reflect the collection policy, and a linear model has limited action/state interactions.

Collection took 85.22 seconds including server lifecycle; held-out evaluation took 41.74 seconds. Thirteen unit tests pass, covering toy learning, exact model reload, observable roster/faint handling, action features, rejection of test-partition training, and the prior legality/CFR tests.

Artifacts retained locally (ignored by Git):

- `runs/experiment-1794850825398704000/plan.json`
- `runs/experiment-1794850825398704000/report.json`
- `runs/experiment-1794850825398704000/outcome.npz`
- `runs/experiment-1794850825398704000/outcome.json`
- `runs/1794850825827973000-local-s11001/` (collection)
- `runs/1794850911734054000-local-s91001/` (evaluation)

Next experiments should use a fresh evaluation seed/pool after development, leaving this result recorded. The promising next model change is nonlinear state/action interactions or reliable simulated successor features, combined with more varied training teams; simply increasing the learned-score weight on this test would reuse the holdout for tuning.
