"""Synthetic AGGREGATE health + workplace signal generator (clearly labelled SYNTHETIC).

The organisers supply no data, so:
  * Environmental signals (WBGT proxy, PM10) are REAL (Open-Meteo, CC BY 4.0) - see fetch_env.py.
  * Clinic, lab, sickness-absence and incident counts are SIMULATED district-day aggregates driven by
    the real weather, plus injected ground-truth events so the detector can be back-tested honestly.
Nothing here is individual-level: the schema has no person, employer-name or employee-id field.
"""
import pathlib
import numpy as np, pandas as pd

DATA = pathlib.Path(__file__).resolve().parent.parent / "data"

# district -> (env zone, covered workers, sector mix weights: construction, industrial, services)
DISTRICTS = {
    "Mussafah":        ("abudhabi", 180_000, (0.55, 0.40, 0.05), (0.50, 0.30)),
    "ICAD":            ("abudhabi",  60_000, (0.10, 0.85, 0.05), (0.52, 0.34)),
    "Khalifa City":    ("abudhabi",  40_000, (0.25, 0.10, 0.65), (0.50, 0.40)),
    "Saadiyat-Yas":    ("abudhabi",  70_000, (0.70, 0.05, 0.25), (0.51, 0.62)),
    "Al Ain Industrial": ("alain",   50_000, (0.30, 0.60, 0.10), (0.78, 0.70)),
    "Al Ain Central":  ("alain",     30_000, (0.15, 0.05, 0.80), (0.80, 0.78)),
    "Ruwais":          ("ruwais",    45_000, (0.15, 0.80, 0.05), (0.22, 0.30)),
    "Al Dhafra":       ("ruwais",    15_000, (0.40, 0.20, 0.40), (0.25, 0.45)),
}
# schematic map positions (x, y in 0..100) for the dashboard; neighbours for spread checks
POS = {"Mussafah": (38, 48), "ICAD": (48, 60), "Khalifa City": (58, 46), "Saadiyat-Yas": (68, 33),
       "Al Ain Industrial": (84, 46), "Al Ain Central": (88, 62), "Ruwais": (14, 40), "Al Dhafra": (30, 62)}
STREAMS = ["heat_visits", "gi_visits", "resp_visits", "lab_gi_pos", "absence", "incidents"]
FAMILY = {"heat_visits": "clinical", "gi_visits": "clinical", "resp_visits": "clinical", "lab_gi_pos": "laboratory",
          "absence": "workplace", "incidents": "workplace"}
# reporting latency in days (data for day d is available at d+lag)
LATENCY = {"heat_visits": 1, "gi_visits": 1, "resp_visits": 1, "lab_gi_pos": 3, "absence": 2, "incidents": 1, "env": 0}

# hazard type -> (affected streams: (multiplier per unit magnitude, lag days))
EVENT_EFFECTS = {
    "heat_cluster":      {"heat_visits": (1.00, 0), "incidents": (0.45, 0), "absence": (0.22, 1)},
    "gi_outbreak":       {"gi_visits": (1.00, 0), "lab_gi_pos": (1.60, 1), "absence": (0.35, 2)},
    "chemical_exposure": {"incidents": (0.90, 0), "resp_visits": (0.80, 0), "absence": (0.30, 1)},
    "resp_cluster":      {"resp_visits": (0.90, 0), "absence": (0.40, 1)},
}
EVENT_SITES = {"heat_cluster": ["Mussafah", "Saadiyat-Yas", "Al Ain Industrial", "Ruwais"],
               "gi_outbreak": ["Mussafah", "Al Ain Industrial", "Al Dhafra", "Saadiyat-Yas", "Ruwais"],
               "chemical_exposure": ["ICAD", "Ruwais", "Mussafah"],
               "resp_cluster": ["Khalifa City", "Al Ain Central", "ICAD", "Mussafah", "Saadiyat-Yas"]}


def load_env():
    env = {}
    for z in ("abudhabi", "alain", "ruwais"):
        env[z] = pd.read_csv(DATA / f"env_{z}.csv", index_col="date", parse_dates=True)
    return env


def make_events(rng, dates, n=64):
    """Ground-truth events spread over 2023-2025 (2022 is a clean burn-in year)."""
    evs, types = [], list(EVENT_EFFECTS)
    lo, hi = pd.Timestamp("2023-01-20"), dates[-1] - pd.Timedelta(days=25)
    span = (hi - lo).days
    slots = np.sort(np.unique((np.linspace(0, span - 1, n * 3) + rng.integers(-6, 7, n * 3)).clip(0, span - 1).astype(int)))
    taken = {}
    for off in slots:
        if len(evs) >= n: break
        t = types[len(evs) % len(types)]
        d = rng.choice(EVENT_SITES[t])
        onset = lo + pd.Timedelta(days=int(off))
        if t == "heat_cluster" and not (4 <= onset.month <= 10):  # heat events need warm season
            continue
        if any(abs((onset - o).days) < 14 for o in taken.get(d, [])):  # no overlap within district
            continue
        taken.setdefault(d, []).append(onset)
        dur = int(rng.integers(7, 15)); ramp = int(rng.integers(3, 6))
        mag = float(rng.choice([0.5, 0.7, 1.0, 1.4, 2.0, 2.8], p=[.10, .16, .24, .22, .18, .10]))
        evs.append(dict(id=f"E{len(evs)+1:02d}", type=t, district=d, onset=onset, duration=dur, ramp=ramp, magnitude=mag))
    return evs


def event_profile(ev, dates):
    """Triangular-ish ramp: rises over `ramp` days to 1, plateaus, decays over last 3 days."""
    k = (dates - ev["onset"]).days.values.astype(float)
    p = np.zeros(len(dates))
    inside = (k >= 0) & (k < ev["duration"])
    up = np.clip((k + 1) / ev["ramp"], 0, 1)
    down = np.clip((ev["duration"] - k) / 3.0, 0, 1)
    p[inside] = np.minimum(up, down)[inside]
    return p


def simulate(seed=7, start="2022-01-01", end="2025-12-31"):
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, end, freq="D")
    env = load_env()
    n = len(dates)
    doy = dates.dayofyear.values; dow = dates.dayofweek.values  # Mon=0 ... Fri=4 (UAE weekend Fri/Sat-Sun mix)
    events = make_events(rng, dates)
    rows = []
    
    for d, (zone, N, mix, (heat_sens, _)) in DISTRICTS.items():
        e = env[zone].reindex(dates).interpolate().bfill().ffill()
        w = e.wbgt_max.values; pm = e.pm10.values
        cons, ind, svc = mix
        heat_exposure = np.maximum(w - 27.0, 0) ** 1.4          # nonlinear heat dose
        eff = {}
        lam_heat = N / 1e5 * (0.10 + 1.15 * (cons + 0.4 * ind)) * (0.35 + 0.16 * heat_exposure) * (0.6 + 0.8 * heat_sens)
        lam_gi = N / 1e5 * 4.6 * (1 + 0.18 * np.sin(2 * np.pi * (doy - 170) / 365)) * np.ones(n)
        lam_resp = N / 1e5 * 5.2 * (1 + 0.30 * np.cos(2 * np.pi * (doy - 20) / 365)) * (1 + 0.0035 * np.maximum(pm - 40, 0))
        lam_lab = lam_gi * 0.34
        lam_abs = N / 1e5 * 95 * (1 + 0.15 * np.cos(2 * np.pi * (doy - 20) / 365)) * np.where(dow == 0, 1.18, np.where(dow == 4, 0.82, 1.0)) * (1 + 0.012 * heat_exposure)
        lam_inc = N / 1e5 * 2.4 * (0.6 + 1.1 * (cons + 0.6 * ind)) * np.where(dow == 4, 0.55, 1.0) * (1 + 0.025 * heat_exposure)
        base = {"heat_visits": lam_heat, "gi_visits": lam_gi, "resp_visits": lam_resp, "lab_gi_pos": lam_lab, "absence": lam_abs, "incidents": lam_inc}
        mult = {s: np.ones(n) for s in STREAMS}
        for ev in (x for x in events if x["district"] == d):
            prof = event_profile(ev, dates)
            for s, (gain, lag) in EVENT_EFFECTS[ev["type"]].items():
                mult[s] = mult[s] + ev["magnitude"] * gain * np.roll(prof, lag) * (np.arange(n) >= lag)
        for s in STREAMS:
            week_eff = np.exp(rng.normal(0, 0.02, n // 7 + 2)[np.arange(n) // 7])
            day_eff = np.exp(rng.normal(0, 0.05, n))
            lam = base[s] * mult[s] * week_eff * day_eff
            lam_expected = base[s] * week_eff * day_eff
            y = rng.poisson(np.maximum(lam, 1e-6))
            rows.append(pd.DataFrame({"date": dates, "district": d, "stream": s, "count": y,
                                      "expected_noevent": lam_expected, "event_excess": np.maximum(lam - lam_expected, 0)}))
        rows.append(pd.DataFrame({"date": dates, "district": d, "stream": "wbgt_max", "count": w, "expected_noevent": w, "event_excess": 0.0}))
        rows.append(pd.DataFrame({"date": dates, "district": d, "stream": "pm10", "count": pm, "expected_noevent": pm, "event_excess": 0.0}))
    long = pd.concat(rows, ignore_index=True)
    ev_df = pd.DataFrame(events)
    return long, ev_df


if __name__ == "__main__":
    long, ev = simulate()
    long.to_parquet(DATA / "signals_long.parquet") if False else long.to_csv(DATA / "signals_long.csv.gz", index=False)
    ev.to_csv(DATA / "events_truth.csv", index=False)
    print(long.shape, len(ev)); print(ev.groupby("type").size().to_dict()); print(ev.head(8).to_string())
