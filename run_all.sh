#!/usr/bin/env bash
# Reproduce everything: real weather -> synthetic signals -> residual engine -> back-test -> dashboard payloads -> tests
set -e
cd "$(dirname "$0")"
python src/fetch_env.py          # Open-Meteo (cached in data/)
python src/simulate.py           # labelled synthetic aggregates + ground-truth events (seeded)
rm -f out/scores.pkl
python src/backtest.py           # calibrate on 2023, freeze, test on 2024-25 (+ baselines, ablations)
python src/robustness.py         # single-feed glitch test
python src/export.py             # role-specific dashboard payloads in web/
python -m pytest tests -q
echo "Serve the dashboard:  cd web && python -m http.server 8765"
