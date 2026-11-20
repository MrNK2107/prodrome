# Prodrome — Technical Documentation

Challenge 06 · Population and Workforce Health Early Warning · Future Health Digital Innovation Challenge

## 1. What it is, in one paragraph

Prodrome is an early-warning layer that sits on top of **aggregate** surveillance feeds. Every day, for each district, it
compares what was reported (clinic visits, lab positives, sickness absence, incident reports) with what the **weather-adjusted
baseline** expected. It accumulates evidence over days, requires **corroboration across materially different signal types**,
holds the false-alert rate to an explicit **alert budget**, and then tells a named human role *what to check, where, and why* —
with the numbers behind it. Outputs are district-level only, role-gated, small-cell-suppressed, and contractually barred from
individual employment or disciplinary use.

## 2. Requirement coverage

| Challenge outcome | How Prodrome meets it | Where |
|---|---|---|
| **Multi-signal surveillance** (≥2 materially different types) | 4 signal families: service (clinic visits ×3), laboratory (GI positives), workplace (absence, incidents), environmental (real work-hours WBGT proxy, PM10). Different latencies (0–3 d) modelled | `simulate.py`, `fetch_env.py` |
| **Explainable anomaly detection** | Per-feed contribution share, observed vs weather-adjusted expected, share of rise explained by weather, hazard signature match, location (district) and period | alert panel; `export.py` |
| **Actionable decision support** | 4 hazards × 3 proportionate tiers (Watch / Advisory / Action) → action text, responsible role, escalation rule, listed assumptions | `actions.py` |
| **Privacy-preserving views** | Aggregate-only schema + guard, k-anonymity small-cell suppression (k=5/10) with complementary suppression, role-specific payloads, purpose limitation | `privacy.py`, `export.py` |
| **Operational validation** | Back-test with lead/delay, sensitivity, PPV, false alarms, latency, missing-input ablations, single-feed glitch test, equal-budget baselines | `backtest.py`, `robustness.py` |
| **Health & resilience impact** | Burden still ahead at alert time → averted illness visits, incidents, lost workdays under an *adjustable, assumed* effectiveness; policy evidence trail | Impact tab |

## 3. Data (and exactly what is real vs simulated)

**No data was supplied by the organisers, so every source is declared:**

| Data | Status | Source / generation |
|---|---|---|
| Work-hours (10:00–16:00) outdoor WBGT proxy, daily max | **Real** | Open-Meteo Archive API (CC BY 4.0), hourly temperature, humidity, shortwave radiation → Stull wet-bulb + crude globe-temperature → WBGT = 0.7·Tw + 0.2·Tg + 0.1·Ta |
| PM10 / PM2.5 daily mean | **Real** | Open-Meteo Air-Quality API (CAMS-based, CC BY 4.0) |
| Three weather zones | Real coordinates | Abu Dhabi/Mussafah (24.35, 54.50), Al Ain (24.21, 55.76), Ruwais/Al Dhafra (24.11, 52.73) |
| Heat-illness, GI, respiratory clinic visits; lab GI positives; sickness-absence episodes; incident reports | **Synthetic** | District-day Poisson counts. Base rates scale with covered workforce and sector mix; heat visits follow a non-linear heat dose of the *real* WBGT; respiratory follows the *real* PM10; plus seasonality, weekday effects, and random-effect overdispersion |
| 64 outbreak/exposure events (16 per hazard: heat cluster, GI outbreak, chemical exposure, respiratory cluster) | **Synthetic** | Injected with known onset, ramp, duration, magnitude and per-feed lags — the ground truth for scoring |
| 8 districts, workforce sizes, sector mixes | **Illustrative** | Plausible emirate-scale districts; not official statistics |

Periods: **2022** burn-in (clean) · **2023** threshold calibration (rolling, out-of-sample) · **2024–25** held-out test (thresholds frozen).
Nothing individual-level exists: the schema has no person, employer-name, ID or GPS field, and `privacy.guard_schema` raises if one appears.

## 4. Architecture

```
Open-Meteo (real)  ─┐
                    ├─►  district × day aggregates ──► [secure aggregate layer]
Clinic / lab /      │      (counts + env)                 1. weather-adjusted baseline (quasi-Poisson GLM, per district×feed)
absence / incident ─┘                                      2. Poisson log-likelihood-ratio CUSUM per feed (leaky, gap-aware)
(synthetic here;                                           3. evidence fusion + corroboration gate + alert budget
 aggregate feeds                                           4. hazard signature → tier → action + role + assumptions
 in production)                                                      │
                                                       role-specific payloads (suppression, granularity)
                                                                     │
                                  dashboard: Public-health · OSH inspectorate · Employer HSE · Policy analyst
```

**Baseline.** For each district × feed: quasi-Poisson GLM on day-of-week, two annual harmonics, heat dose `max(WBGT−27,0)^1.4`
and dust `max(PM10−40,0)/50`. Refit every 14 days on the trailing 365 days; a second pass down-weights upper-tail residual days
so past outbreaks do not contaminate the baseline. Residuals are well calibrated (standardised-residual sd ≈ 1.0–1.2 across feeds).

**Evidence.** Per feed, a leaky Poisson-LLR CUSUM tuned to an ~80 % rise: `S_t = min(cap, max(0, 0.9·S_{t-1} + (y·ln(1+δ) − μδ)/φ))`.
Log-likelihoods add across independent feeds, so the sum is a principled fused evidence score. Missing days **decay** the
statistic; they never create evidence.

**Latency.** Each feed is delayed by its reporting lag (clinic 1 d, lab 3 d, absence 2 d, incidents 1 d, environment 0 d), so the
alert on day *t* uses only data actually available on day *t*.

**Corroboration gate.** An alert opens when fused evidence > θ **and** ≥2 feeds each carry ≥ρ·θ evidence from ≥2 distinct
families (service / laboratory / workplace). One feed, however loud, cannot alert alone.

**Alert-fatigue calibration.** θ is the smallest value keeping system-wide false alerts ≤ **1 per 30 days** on the 2023 calibration year.
ρ is chosen on 2023 only, subject to a design constraint that ≤30 % of injected single-feed glitches may raise an alert
(result: θ = 6.0, ρ = 0.35). Both are then **frozen** for 2024–25. Alerts close after 3 days below θ/2, then a 5-day cool-down.

**Explainability.** At alert open: share of evidence by feed; observed vs expected over the last 3 available days; weather-explained
share = (expected_with_weather − expected_weather_blind) / (observed − expected_weather_blind); hazard = cosine match of the
contribution vector to four clinical signatures (+ WBGT context). 21 of 22 true-positive alerts were classified to the right hazard.

**Proportionate action.** Tier by evidence strength and corroboration: *Watch* (monitor), *Advisory* (targeted, low-cost verification
in the district), *Action* (≥3 feeds and evidence ≥2.2·θ). Each playbook line names a **role** (e.g. OSH inspectorate, public-health
surveillance officer, environmental-health officer), never an individual or employer. A human decides and verifies every alert.

## 5. Privacy and governance

- **Aggregate-only** schema, enforced in code and tests.
- **Small-cell suppression**: any displayed count 1…k−1 replaced by "<k" (k=5 operational roles, k=10 broad roles) + complementary suppression.
- **Role-based payloads**: each role's JSON contains only what it may see — enforced by construction (the browser never receives hidden data):

| Role | Granularity | k | Counts | Evidence & actions |
|---|---|---|---|---|
| Public-health surveillance officer | district-day | 5 | yes (6 feeds) | yes |
| OSH / labour inspectorate | district-day | 5 | yes (4 feeds) | yes |
| Employer HSE lead | district-week | 10 | bands only | no |
| Policy / regulatory analyst | district-month | 10 | bands only | no |

- **Purpose limitation** is displayed on every screen: aggregate surveillance only; not for individual employment, disciplinary,
  immigration, visa or insurance decisions; never identifies workers or employers.
- **Geolocation**: operates at district level; no precise coordinates are needed, so incomplete geocoding does not break it.
- Production would add server-side authN/Z, audit logging, DPIA/clinical-governance sign-off and data-sharing agreements.

## 6. Operational validation

Protocol: 32 held-out events (2024–25), thresholds frozen from 2023. A **detection** = alert in the event's district within
[onset−2 d, end+3 d]; any other alert is a **false alert**. Baselines are calibrated to the same alert budget on 2023.

| Method | Sens. | PPV | False alerts/30 d | Median delay | Event share still ahead |
|---|---|---|---|---|---|
| **Prodrome** | **69 % (22/32)** | **73 %** | **0.33** | 5 d | 48 % |
| Prodrome without corroboration gate | 78 % | 48 % | 1.11 | 4 d | 57 % |
| Daily single-feed threshold | 28 % | 32 % | 0.78 | 4 d | 50 % |
| Weekly retrospective review | 44 % | 29 % | 1.52 | 8 d | 15 % |
| Un-calibrated "3σ on any feed" | 81 % | 12 % | 9.6 | – | – |

**By hazard:** chemical exposure 8/8 · respiratory 7/8 · GI 5/8 · **heat 2/8**. **By strength:** strong (≥2.0) 10/11 · moderate 9/12 · weak 3/9.
Misses concentrate in weak signals in small-count districts (e.g. 4 observed vs 3 expected visits) — a minimum-detectable-effect limit.
Heat is the weakest hazard here because its baseline is already strongly weather-driven, so only large excess is distinguishable.

**Lead time.** Prodrome alerts a median 5 d after onset, **3 d earlier than weekly review**, leaving a median 48 % of an event's
duration ahead (vs 15 %). Alerts typically arrive during the ramp, after the peak day for short ramps — the claim is *earlier than today's
practice*, not *before onset*.

**Data latency.** Doubling all reporting lags leaves sensitivity at 69 % but delays median detection from 5 to 7 days and raises false alerts to 0.41/30 d.

**Missing inputs (same frozen thresholds):**

| Scenario | Sens. | PPV | FA/30 d | Delay |
|---|---|---|---|---|
| All feeds | 69 % | 73 % | 0.33 | 5.0 |
| Lab feed offline | 72 % | 88 % | 0.12 | 5.0 |
| Sickness-absence feed offline | 50 % | 67 % | 0.33 | 6.0 |
| Incident feed offline | 62 % | 83 % | 0.16 | 5.5 |
| Environmental feed offline (no weather adjustment) | 72 % | 74 % | 0.33 | 5.0 |
| 20 % random daily gaps in every feed | 62 % | 77 % | 0.25 | 5.0 |
| Reporting latency doubled | 69 % | 69 % | 0.41 | 7.0 |
| All clinical + lab feeds offline | 0 % | – | 0 | fails **silent** by design |

Some rows rise in PPV when a feed is dropped because that feed also contributed false alerts — small-sample noise (32 events), not advice.

**Faulty-feed robustness.** 5-day ×3 spike injected into a single feed (120 trials, no real event): **84 %** would alert without
corroboration; **26 %** do with it (design limit 30 %, set on 2023).

## 7. Impact (simulated, assumption-driven)

Burden = injected event excess over the test period: 914 illness visits, 339 incidents, 7,323 absence episodes (≈ lost workdays).
Burden still ahead when the first alert fires: **Prodrome 320 / 122 / 3,344** vs weekly review **69 / 17 / 807**.
Under an *assumed* 30 % effectiveness of timely action: ≈ **96 visits, 37 incidents and ~1,000 workdays averted** vs ≈ 21 / 5 / 242 for weekly review.
The effectiveness slider in the dashboard shows sensitivity. These are illustrations of the mechanism, not forecasts.
Policy value: each alert stores district, window, feeds, expected-vs-observed and action — an auditable trail for inspection programmes and regulation.

## 8. Clinical boundaries

Prodrome is population-level surveillance and decision support. It does **not** diagnose, triage or monitor individuals, and it never
recommends action against a worker or employer. Alerts are hypotheses to verify in the field by a named public-health or OSH professional.
Hazard labels are signatures, not diagnoses.

## 9. Limitations and honest caveats

1. Health, absence and incident data are **synthetic**; effect sizes are chosen by the author. Real performance will differ and needs a pilot.
2. The weather→illness link is **modelled**, not observed; real data will carry reporting artefacts, weekday/holiday patterns, and population churn this simulation under-represents.
3. WBGT is a proxy (no wind, simplified globe temperature); PM10 is reanalysis, not station data.
4. District-level resolution only; no sub-district spatial scan statistic; cross-district spread is not modelled.
5. 32 test events → wide confidence intervals; hazard-level numbers (n = 8) are indicative only.
6. Heat clusters are the weakest detected hazard (2/8); weak signals in small districts are below the minimum detectable effect.
7. Calibration assumes the 2023 noise level persists; drift monitoring and periodic recalibration would be required.
8. Deployment would need data-sharing agreements, DPIA, clinical governance, and server-side access control (the dashboard demonstrates role payloads; it is not an authenticated service).

## 10. Path to pilot

1. Choose 2–3 districts and 2 feeds already collected centrally (e.g. heat-illness ED/clinic visits + sickness-absence aggregates).
2. Run silently for 8–12 weeks, back-testing against retrospective events identified by the public-health team.
3. Tune θ to the receiving team's capacity (alerts per week they can genuinely act on).
4. Go live with *Watch/Advisory* tiers only; add *Action* tier after review. Track alert-to-action rate, precision, and time-to-verification.

## 11. Reproducibility

`bash run_all.sh` regenerates everything (seeded). 11 automated tests cover small-cell suppression, complementary suppression,
schema guard, role-payload entitlements, gap-aware CUSUM, corroboration gate, no-future-leak causality, latency shift, and back-test consistency.
Environmental data © Open-Meteo, CC BY 4.0.
