"""Back-test Prodrome against injected ground-truth events + baselines + missing-data stress tests.
Periods: 2022 burn-in | 2023 = threshold calibration (out-of-sample rolling) | 2024-25 = FROZEN-threshold test."""
import json, pickle, pathlib, sys, time, zlib
from concurrent.futures import ProcessPoolExecutor
import numpy as np, pandas as pd
from simulate import STREAMS, FAMILY, LATENCY, DISTRICTS, POS
from detect import residual_streams, stream_scores, fuse, alerts_from_fused, ACTIVE_TAU
import actions as A

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA, OUT = ROOT / "data", ROOT / "out"
CAL = ("2023-01-01", "2023-12-31"); TEST = ("2024-01-01", "2025-12-31")
BUDGET_FA_PER_30D = 1.0         # system-wide false alerts per 30 days (alert-fatigue budget)
TOL_BEFORE, TOL_AFTER = 2, 3     # alert counts as matching an event if opened within [onset-2, end+3]
SEED_MISS = 11


def _district_job(args):
    d, variant = args
    long = pd.read_csv(DATA / "signals_long.csv.gz")
    dates = pd.date_range(long.date.min(), long.date.max(), freq="D")
    kw = {}
    if variant == "noenv":
        kw["drop_env_cols"] = True
    elif variant == "miss20":
        rng = np.random.default_rng(SEED_MISS + zlib.crc32(d.encode()) % 1000)
        kw["missing_mask"] = {s: rng.random(len(dates)) < 0.20 for s in STREAMS}
    dts, X, out = residual_streams(long, d, **kw)
    sc = stream_scores(dts, out)
    slim = {s: {k: v for k, v in o.items()} for s, o in sc.items()}
    env = long[(long.district == d) & (long.stream == "wbgt_max")].set_index(pd.to_datetime(long[(long.district == d) & (long.stream == "wbgt_max")].date))["count"].reindex(dts).values
    pm = long[(long.district == d) & (long.stream == "pm10")].set_index(pd.to_datetime(long[(long.district == d) & (long.stream == "pm10")].date))["count"].reindex(dts).values
    return d, variant, dict(dates=dts, scores=slim, wbgt=env, pm=pm)


def compute_all(variants=("full", "noenv", "miss20")):
    cache = OUT / "scores.pkl"
    if cache.exists():
        return pickle.load(open(cache, "rb"))
    jobs = [(d, v) for v in variants for d in DISTRICTS]
    res = {}
    with ProcessPoolExecutor(max_workers=4) as ex:
        for d, v, r in ex.map(_district_job, jobs):
            res[(d, v)] = r; print("done", d, v, flush=True)
    pickle.dump(res, open(cache, "wb"))
    return res


def period_mask(dates, p):
    return (dates >= pd.Timestamp(p[0])) & (dates <= pd.Timestamp(p[1]))


def match(alerts, ev, dates):
    """alerts: list of dict(district, open_idx). Returns per-event detected/first alert, FA list."""
    used = set(); per_ev = {}
    for e in ev.itertuples():
        lo = e.onset - pd.Timedelta(days=TOL_BEFORE); hi = e.onset + pd.Timedelta(days=e.duration + TOL_AFTER)
        hits = [i for i, a in enumerate(alerts) if a["district"] == e.district and lo <= dates[a["open_idx"]] <= hi]
        per_ev[e.id] = hits
        used.update(hits)
    fa = [i for i in range(len(alerts)) if i not in used]
    return per_ev, fa



GLITCH_MAX = 0.30   # design constraint: at most 30% of single-feed artefacts may raise an alert


def glitch_rate(res, ev, theta, rho, window, n_trials=80, seed=5, gate_on=True):
    """Inject a 5-day x3 spike into ONE feed of ONE district (no real event nearby) and count alerts within 12 days."""
    import copy
    from detect import cusum
    dts = res[("Mussafah", "full")]["dates"]; n = len(dts)
    lo = dts.get_loc(pd.Timestamp(window[0])); hi = dts.get_loc(pd.Timestamp(window[1]))
    lo = max(lo, 320); rng = np.random.default_rng(seed); ds = list(DISTRICTS); hits = N = 0
    while N < n_trials:
        d = ds[rng.integers(len(ds))]; s = STREAMS[rng.integers(len(STREAMS))]; st = int(rng.integers(lo + 5, hi - 15))
        if any(e.district == d and abs((dts[st] - e.onset).days) < 25 for e in ev.itertuples()): continue
        sc = copy.deepcopy(res[(d, "full")]["scores"]); o = sc[s]; y = o["y"].copy()
        y[st:st + 5] = np.round(y[st:st + 5] * 3 + rng.poisson(2, 5))
        sc[s]["cusum"] = cusum(y, o["mu"], o["phi"], o["avail"])
        F, act, gate, _ = fuse(sc, theta, n=n, rho=rho)
        g = gate if gate_on else np.ones(n, bool)
        hits += int(any(st <= a[0] < st + 12 for a in alerts_from_fused(dts, F, g, theta))); N += 1
    return hits / N


def run_variant(res, variant, ev, theta=None, rho=None, streams=None, latency=None, gate_on=True, window=TEST, calibrate=True, budget=BUDGET_FA_PER_30D):
    """Fuse for every district; calibrate (theta, rho) on CAL if not given; evaluate on `window`."""
    def build(th, rh):
        al = []; fused = {}
        for d in DISTRICTS:
            r = res[(d, variant)]; dts = r["dates"]; n = len(dts)
            F, act, gate, contrib = fuse(r["scores"], th, streams=streams, latency=latency, n=n, rho=rh)
            g = gate if gate_on else np.ones(n, bool)
            fused[d] = (F, act, gate, contrib)
            for (i, j) in alerts_from_fused(dts, F, g, th):
                al.append(dict(district=d, open_idx=i, close_idx=j))
        return al, fused
    dts = res[(next(iter(DISTRICTS)), variant)]["dates"]
    if theta is None and calibrate:
        cm = period_mask(dts, CAL); cal_ev = ev[(ev.onset >= pd.Timestamp(CAL[0])) & (ev.onset <= pd.Timestamp(CAL[1]))]
        best = None
        for rh in ((0.25,) if not gate_on else (0.15, 0.25, 0.35, 0.5)):
            for th in np.arange(2.0, 30.0, 0.5):
                al, _ = build(th, rh); al = [a for a in al if cm[a["open_idx"]]]
                per, fa = match(al, cal_ev, dts)
                if len(fa) / (cm.sum() / 30.0) <= budget:
                    sens = sum(1 for v in per.values() if v) / max(len(cal_ev), 1)
                    gl = glitch_rate(res, ev, th, rh, CAL) if gate_on else 0.0
                    key = (gl <= GLITCH_MAX, sens, -th)
                    if best is None or key > best[0]: best = (key, th, rh)
                    break   # smallest feasible theta for this rho
        theta, rho = (best[1], best[2]) if best else (30.0, 0.25)
    rho = 0.25 if rho is None else rho
    al, fused = build(theta, rho)
    m = period_mask(dts, window); al_t = [a for a in al if m[a["open_idx"]]]
    ev_t = ev[(ev.onset >= pd.Timestamp(window[0])) & (ev.onset <= pd.Timestamp(window[1]))]
    return (theta, rho), al_t, ev_t, fused, dts


def metrics(al, ev_t, dts, window, long_events_excess=None):
    per_ev, fa = match(al, ev_t, dts)
    det = [e for e in ev_t.itertuples() if per_ev[e.id]]
    tp_alerts = len(al) - len(fa)
    leads, delays, preventable = [], [], []
    for e in det:
        first = min(per_ev[e.id], key=lambda i: al[i]["open_idx"])
        ad = dts[al[first]["open_idx"]]
        delays.append((ad - e.onset).days)
        peak = e.onset + pd.Timedelta(days=e.ramp - 1)
        leads.append((peak - ad).days)
        end = e.onset + pd.Timedelta(days=e.duration)
        tot = max((end - e.onset).days, 1)
        after = max((end - max(ad, e.onset)).days, 0)
        preventable.append(after / tot)
    ndays = (pd.Timestamp(window[1]) - pd.Timestamp(window[0])).days + 1
    first_alert = {e.id: dts[al[min(per_ev[e.id], key=lambda i: al[i]["open_idx"])]["open_idx"]] for e in det}
    return dict(first_alert=first_alert, events=len(ev_t), detected=len(det), sensitivity=len(det) / max(len(ev_t), 1),
                alerts=len(al), false_alerts=len(fa), ppv=tp_alerts / max(len(al), 1),
                fa_per_30d=len(fa) / (ndays / 30.0),
                median_detection_delay_days=float(np.median(delays)) if delays else None,
                median_lead_before_peak_days=float(np.median(leads)) if leads else None,
                median_event_share_after_alert=float(np.median(preventable)) if preventable else None,
                detected_ids=[e.id for e in det], missed_ids=[e.id for e in ev_t.itertuples() if not per_ev[e.id]])


# ---------------- baselines ----------------
def raw_scores_for(long, d):
    """Naive per-stream z against trailing 28-day mean/sd (no weather adjustment, no fusion)."""
    sub = long[long.district == d]; out = {}
    for s in STREAMS:
        y = sub[sub.stream == s].sort_values("date")["count"].values.astype(float)
        ser = pd.Series(y)
        mu = ser.shift(1).rolling(28).mean(); sd = ser.shift(1).rolling(28).std().clip(lower=0.7)
        z = ((ser - mu) / sd).values
        lag = LATENCY[s]
        out[s] = np.concatenate([np.full(lag, np.nan), z[: len(z) - lag]]) if lag else z
    return out


def weekly_scores_for(long, d):
    """Weekly retrospective single-stream review: Monday review of last full week vs trailing 8 weeks."""
    sub = long[long.district == d]; out = {}
    for s in STREAMS:
        ser = sub[sub.stream == s].sort_values("date").set_index(pd.to_datetime(sub[sub.stream == s].sort_values("date").date))["count"]
        w = ser.resample("W-SUN").sum()
        mu = w.shift(1).rolling(8).mean(); sd = w.shift(1).rolling(8).std().clip(lower=1.0)
        z = ((w - mu) / sd)
        full = z.reindex(ser.index, method=None)
        # available from the Monday after the week ends, plus reporting lag
        s_arr = np.full(len(ser), np.nan)
        for wk_end, val in z.items():
            idx = ser.index.get_indexer([wk_end + pd.Timedelta(days=1 + LATENCY[s])])[0]
            if idx >= 0: s_arr[idx] = val
        out[s] = s_arr
    return out


def baseline_eval(long, ev, dts, scorer, budget=BUDGET_FA_PER_30D, name="baseline"):
    sc = {d: scorer(long, d) for d in DISTRICTS}
    def build(th):
        al = []
        for d in DISTRICTS:
            m = np.nanmax(np.vstack([sc[d][s] for s in STREAMS]), axis=0)
            m = np.nan_to_num(m, nan=-9)
            i = 0
            while i < len(m):
                if m[i] > th:
                    al.append(dict(district=d, open_idx=i)); i += 7
                else: i += 1
        return al
    cm = period_mask(dts, CAL); cal_ev = ev[(ev.onset >= pd.Timestamp(CAL[0])) & (ev.onset <= pd.Timestamp(CAL[1]))]
    th_sel = 12.0
    for th in np.arange(1.5, 12.0, 0.1):
        al = [a for a in build(th) if cm[a["open_idx"]]]
        _, fa = match(al, cal_ev, dts)
        if len(fa) / (cm.sum() / 30.0) <= budget:
            th_sel = float(th); break
    tm = period_mask(dts, TEST); al = [a for a in build(th_sel) if tm[a["open_idx"]]]
    ev_t = ev[(ev.onset >= pd.Timestamp(TEST[0])) & (ev.onset <= pd.Timestamp(TEST[1]))]
    m = metrics(al, ev_t, dts, TEST); m["threshold"] = th_sel; m["name"] = name
    # also report the uncalibrated "classic" 3-sigma alert burden
    al3 = [a for a in build(3.0) if tm[a["open_idx"]]]; _, fa3 = match(al3, ev_t, dts)
    m["classic_3sigma"] = dict(alerts=len(al3), false_alerts=len(fa3), fa_per_30d=len(fa3) / (((pd.Timestamp(TEST[1]) - pd.Timestamp(TEST[0])).days + 1) / 30.0),
                               sensitivity=len(set(e.id for e in ev_t.itertuples() if any(a["district"] == e.district and e.onset - pd.Timedelta(days=TOL_BEFORE) <= dts[a["open_idx"]] <= e.onset + pd.Timedelta(days=e.duration + TOL_AFTER) for a in al3))) / max(len(ev_t), 1))
    return m


def jsonable(o):
    if isinstance(o, dict): return {k: jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [jsonable(v) for v in o]
    if isinstance(o, (np.floating,)): return float(o)
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, pd.Timestamp): return o.strftime("%Y-%m-%d")
    return o


def main():
    t0 = time.time()
    long = pd.read_csv(DATA / "signals_long.csv.gz")
    ev = pd.read_csv(DATA / "events_truth.csv", parse_dates=["onset"])
    res = compute_all()
    print("scores ready", round(time.time() - t0), "s", flush=True)
    dts = res[("Mussafah", "full")]["dates"]
    R = {}
    (theta, rho), al, ev_t, fused, _ = run_variant(res, "full", ev)
    R["prodrome"] = metrics(al, ev_t, dts, TEST); R["prodrome"]["theta"] = theta; R["prodrome"]["rho"] = rho; R["prodrome"]["name"] = "Prodrome (fused, weather-adjusted)"
    # no-gating ablation (alert on fused score alone, same budget calibration)
    (th2, _r2), al2, _, _, _ = run_variant(res, "full", ev, gate_on=False)
    R["no_corroboration_gate"] = metrics(al2, ev_t, dts, TEST); R["no_corroboration_gate"]["theta"] = th2
    # baselines
    R["daily_single_stream"] = baseline_eval(long, ev, dts, raw_scores_for, name="Daily single-stream threshold (no weather adj, no fusion)")
    R["weekly_retrospective"] = baseline_eval(long, ev, dts, weekly_scores_for, name="Weekly retrospective single-stream review")
    # missing-input behaviour (frozen theta from full model; honest degradation)
    abl = {}
    def frozen(variant, streams=None, latency=None, label=""):
        _, a, _, _, _ = run_variant(res, variant, ev, theta=theta, rho=rho, streams=streams, latency=latency)
        m = metrics(a, ev_t, dts, TEST); abl[label] = {k: m[k] for k in ("sensitivity", "ppv", "fa_per_30d", "median_detection_delay_days", "median_lead_before_peak_days", "detected", "events", "alerts", "false_alerts")}
    frozen("full", label="Baseline (all feeds)")
    frozen("full", streams=[s for s in STREAMS if s != "lab_gi_pos"], label="Lab feed offline")
    frozen("full", streams=[s for s in STREAMS if s != "absence"], label="Sickness-absence feed offline")
    frozen("full", streams=[s for s in STREAMS if s != "incidents"], label="Incident-report feed offline")
    frozen("full", streams=[s for s in STREAMS if FAMILY[s] == "workplace"], label="All clinical + lab feeds offline (workplace only)")
    frozen("noenv", label="Environmental feed offline (no weather adjustment)")
    frozen("miss20", label="20% random daily data gaps in every feed")
    frozen("full", latency={k: v * 2 for k, v in LATENCY.items()}, label="Reporting latency doubled")
    R["missing_input"] = abl
    # strata
    strata = {"type": {}, "magnitude": {}}
    for t, g in ev_t.groupby("type"):
        strata["type"][t] = dict(n=len(g), detected=sum(e in R["prodrome"]["detected_ids"] for e in g.id))
    bins = [(0, 0.8, "weak (<=0.7)"), (0.8, 1.5, "moderate (1.0-1.4)"), (1.5, 9, "strong (>=2.0)")]
    for lo, hi, nm in bins:
        g = ev_t[(ev_t.magnitude >= lo) & (ev_t.magnitude < hi)]
        strata["magnitude"][nm] = dict(n=len(g), detected=sum(e in R["prodrome"]["detected_ids"] for e in g.id))
    R["strata"] = strata
    R["config"] = dict(test_period=TEST, calibration_period=CAL, alert_budget_fa_per_30d=BUDGET_FA_PER_30D, n_districts=len(DISTRICTS),
                       latency=LATENCY, active_tau=ACTIVE_TAU, match_tolerance_days=[TOL_BEFORE, TOL_AFTER])
    json.dump(jsonable(R), open(OUT / "backtest.json", "w"), indent=2)
    pickle.dump(dict(theta=theta, rho=rho), open(OUT / "theta.pkl", "wb"))
    print(json.dumps(jsonable({k: ({kk: vv for kk, vv in v.items() if kk not in ("detected_ids", "missed_ids")} if isinstance(v, dict) and k in ("prodrome", "no_corroboration_gate", "daily_single_stream", "weekly_retrospective") else v) for k, v in R.items() if k != "config"}), indent=1))
    print("total", round(time.time() - t0), "s")


if __name__ == "__main__":
    main()
