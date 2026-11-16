#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.lock
.venv/bin/python -m pip install -e .
if [ ! -d vendor/pokemon-showdown ]; then
  git clone https://github.com/smogon/pokemon-showdown.git vendor/pokemon-showdown
fi
revision=$(cat configs/showdown-revision.txt)
git -C vendor/pokemon-showdown checkout "$revision"
(cd vendor/pokemon-showdown && npm ci --omit=optional && node build)
node scripts/validate-team.cjs
.venv/bin/python -m pytest -q
