# Prodrome — Population & Workforce Health Early Warning

**Future Health Digital Innovation Challenge · Challenge 06 · Population Health**

Prodrome fuses clinic, laboratory, workplace (sickness absence, incident reports) and environmental (heat, dust)
signals at district level to raise an **explainable, privacy-preserving, alert-budgeted** early warning — and
recommends a *proportionate* preventive action and the *responsible role*, with the evidence and assumptions shown.

- **Live dashboard:** https://mrnk2107.github.io/prodrome/ (no login; pick a role under "View as")
- **Demo video (2:51):** https://mrnk2107.github.io/prodrome/demo.mp4
- **Submission deck:** [`submission/Prodrome_FutureHealth_Submission.pdf`](submission/Prodrome_FutureHealth_Submission.pdf)
- Run locally: `cd web && python -m http.server 8765`, then open http://localhost:8765
- Full technical documentation: [`docs/TECHNICAL_DOCUMENTATION.md`](docs/TECHNICAL_DOCUMENTATION.md)
- Reproduce all numbers: `bash run_all.sh` (≈1 min; needs internet once for Open-Meteo).

## Headline results (held-out 2024–25 test, thresholds frozen from 2023)
| Method (same alert budget, calibrated on 2023) | Sensitivity | PPV | False alerts / 30 d | Median delay |
|---|---|---|---|---|
| **Prodrome (fused, weather-adjusted, corroborated)** | **69 %** (22/32) | **73 %** | **0.33** | 5 d |
| Daily single-feed threshold | 28 % | 32 % | 0.78 | 4 d |
| Weekly retrospective review (today's practice) | 44 % | 29 % | 1.52 | 8 d |

Strong events: 10/11 caught. Single-feed data glitches that raise an alert: **26 %** with corroboration vs 84 % without.
An un-calibrated "3-sigma on any feed" rule raises **9.6 false alerts per 30 days**. All events are **simulated** — see limitations.

## Honest scope
Environmental inputs are **real** (Open-Meteo weather & air quality, CC BY 4.0). Clinic/lab/absence/incident counts are
**synthetic aggregates** with injected, ground-truthed events, because the organisers supply no data. The value demonstrated
is the method, the equal-budget comparison and the stress tests — not absolute real-world accuracy.

## Layout
`src/` pipeline (fetch_env, simulate, detect, actions, privacy, backtest, robustness, export) · `web/` dashboard + role payloads ·
`tests/` 11 tests · `out/backtest.json` results · `docs/` documentation
