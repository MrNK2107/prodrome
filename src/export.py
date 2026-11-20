"""Export dashboard payloads. One JSON per ROLE: each contains only the fields/granularity/suppression that role is
entitled to, so privacy is enforced by construction (the client never receives data it must hide)."""
import json, pickle, pathlib
import numpy as np, pandas as pd
from simulate import STREAMS, FAMILY, LATENCY, DISTRICTS, POS
import backtest as BT
import actions as A
import privacy as P

ROOT = pathlib.Path(__file__).resolve().parent.parent
WEB = ROOT / "web"
NAMES = {"heat_visits": "Heat-illness clinic visits", "gi_visits": "GI clinic visits", "resp_visits": "Respiratory clinic visits",
         "lab_gi_pos": "Lab-confirmed GI positives", "absence": "Sickness-absence episodes", "incidents": "Workplace incident reports"}
STREAM_GROUP = {"heat_visits": "Clinical service", "gi_visits": "Clinical service", "resp_visits": "Clinical service",
                "lab_gi_pos": "Laboratory", "absence": "Workplace", "incidents": "Workplace"}
WEATHER_SENS = {"heat_visits", "resp_visits", "incidents", "absence"}


def r1(x): return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), 1)


def main():
    long = pd.read_csv(BT.DATA / "signals_long.csv.gz"); ev = pd.read_csv(BT.DATA / "events_truth.csv", parse_dates=["onset"])
    res = pickle.load(open(BT.OUT / "scores.pkl", "rb"))
    dts = res[("Mussafah", "full")]["dates"]
    (theta, rho), al, ev_t, fused, _ = BT.run_variant(res, "full", ev)
    per, fa = BT.match(al, ev_t, dts)
    ev_by = {e.id: e for e in ev_t.itertuples()}
    t0 = dts.get_loc(pd.Timestamp(BT.TEST[0])); idx = np.arange(t0, len(dts))
    dates = [dts[i].strftime("%Y-%m-%d") for i in idx]
    P.guard_schema(long.columns)

    # ---- alerts with explanations ----
    alerts = []
    matched_ev = {i: e for e, hits in per.items() for i in hits}
    for k, a in enumerate(sorted(al, key=lambda x: x["open_idx"])):
        d = a["district"]; t = a["open_idx"]
        F, act, gate, contrib = fused[d]
        win = slice(max(t - 1, 0), t + 1)
        cvec = {s: float(np.max(contrib[s][win])) for s in STREAMS}
        wb = res[(d, "full")]["wbgt"]; wbgt_hi = bool(np.nanmax(wb[max(t - 3, 0):t + 1]) >= 31.0)
        hazard, sims, margin = A.classify(cvec, wbgt_hi)
        peakF = float(np.max(F[t:a["close_idx"] + 1])); n_act = int(act[t])
        tr = A.tier(F[t], theta, n_act)            # tier AS OF the opening day (no lookahead)
        timeline = []; prev = None                 # tier may escalate / de-escalate as evidence evolves (as-of each day)
        for dd in range(t, a["close_idx"] + 1):
            ti = A.tier(F[dd], theta, int(act[dd]))
            if ti != prev: timeline.append([int(dd - t0), ti]); prev = ti
        playbook = {k: dict(action=A.PLAYBOOK[hazard][k][0], role=A.PLAYBOOK[hazard][k][1]) for k in A.PLAYBOOK[hazard]}
        action, role = A.recommend(hazard, tr)
        tot = sum(cvec.values()) + 1e-9
        ev_lines = []
        for s in sorted(STREAMS, key=lambda s: -cvec[s])[:3]:
            if cvec[s] < 0.08 * tot: continue
            lag = LATENCY[s]; hi = t - lag; lo = hi - 2
            f = res[(d, "full")]["scores"][s]; fn = res[(d, "noenv")]["scores"][s]
            obs = float(np.nansum(f["y"][lo:hi + 1])); exp = float(np.nansum(f["mu"][lo:hi + 1])); exp_blind = float(np.nansum(fn["mu"][lo:hi + 1]))
            wx = None
            if s in WEATHER_SENS and obs - exp_blind > 1:
                wx = float(np.clip((exp - exp_blind) / (obs - exp_blind), 0, 1))
            ev_lines.append(dict(stream=s, name=NAMES[s], group=STREAM_GROUP[s], share=cvec[s] / tot, observed=obs, expected=exp,
                                 ratio=obs / max(exp, 0.5), weather_explained=wx, lag_days=lag, window=[dts[lo].strftime("%Y-%m-%d"), dts[hi].strftime("%Y-%m-%d")]))
        truth = None
        if t in {x["open_idx"] for x in al} and matched_ev.get(al.index(a)) is not None:
            e = ev_by[matched_ev[al.index(a)]]
            truth = dict(id=e.id, type=e.type, onset=e.onset.strftime("%Y-%m-%d"), magnitude=e.magnitude, hazard_correct=(A.TRUTH_MAP[e.type] == hazard))
        alerts.append(dict(id=f"A{k+1:03d}", district=d, open=dts[t].strftime("%Y-%m-%d"), close=dts[a["close_idx"]].strftime("%Y-%m-%d"),
                           open_i=int(t - t0), close_i=int(a["close_idx"] - t0), peak_score=r1(peakF), score=r1(F[t]), theta=theta, tier=tr, tiers=timeline, playbook=playbook,
                           hazard=hazard, hazard_label=A.LABEL[hazard], hazard_margin=round(margin, 2), active_streams=n_act,
                           contrib={s: round(cvec[s] / tot, 3) for s in STREAMS}, evidence=ev_lines, action=action, role=role,
                           wbgt_hi=wbgt_hi, truth=truth))
    for al_ in alerts:
        al_["status"] = "true_positive" if al_["truth"] else "false_alarm"

    # ---- impact: burden after alert (prodrome vs weekly review) ----
    ex = long[long.stream.isin(STREAMS)].copy(); ex["date"] = pd.to_datetime(ex["date"])
    BASE = json.load(open(BT.OUT / "backtest.json"))
    R_p = BT.metrics(al, ev_t, dts, BT.TEST)
    wk = BT.baseline_eval(long, ev, dts, BT.weekly_scores_for, name="weekly")
    def burden(first_alert):
        tot = dict(visits=0.0, incidents=0.0, workdays=0.0); after = dict(visits=0.0, incidents=0.0, workdays=0.0)
        for e in ev_t.itertuples():
            g = ex[(ex.district == e.district) & (ex.date >= e.onset - pd.Timedelta(days=1)) & (ex.date <= e.onset + pd.Timedelta(days=e.duration + 4))]
            fa_date = first_alert.get(e.id)
            for s, grp in (("heat_visits", "visits"), ("gi_visits", "visits"), ("resp_visits", "visits"), ("incidents", "incidents"), ("absence", "workdays")):
                gs = g[g.stream == s]
                tot[grp] += float(gs.event_excess.sum())
                if fa_date is not None:
                    after[grp] += float(gs[gs.date > fa_date].event_excess.sum())
        return tot, after
    tot, after_p = burden(R_p["first_alert"]); _, after_w = burden(wk["first_alert"])
    impact = dict(total_excess=tot, after_prodrome=after_p, after_weekly=after_w, default_effectiveness=0.30,
                  note="Excess burden = simulated event_excess (known by construction). Averted = post-alert excess x assumed intervention effectiveness (assumption, adjustable).")

    # ---- role payloads ----
    sub = long[long.stream.isin(STREAMS + ["wbgt_max", "pm10"])].copy(); sub["date"] = pd.to_datetime(sub["date"])
    piv = {d: {s: sub[(sub.district == d) & (sub.stream == s)].set_index("date")["count"].reindex(dts).values for s in STREAMS + ["wbgt_max", "pm10"]} for d in DISTRICTS}
    mu = {d: {s: res[(d, "full")]["scores"][s]["mu"] for s in STREAMS} for d in DISTRICTS}
    districts = [dict(name=d, pos=POS[d], workers=DISTRICTS[d][1], zone=DISTRICTS[d][0]) for d in DISTRICTS]
    common = dict(dates=dates, districts=districts, theta=theta, rho=rho, purpose=P.PURPOSE_STATEMENT, stream_names=NAMES, stream_group=STREAM_GROUP,
                  latency=LATENCY, generated="synthetic health/workplace signals + real Open-Meteo weather/air quality")

    def agg_weeks(arr, k):  # weekly sums over test window
        s = pd.Series(arr[idx], index=dts[idx]); w = s.resample("W-MON", label="left", closed="left").sum()
        return [dts_.strftime("%Y-%m-%d") for dts_ in w.index], w.values
    def risk_band(sc):  # coarse band, no counts
        return "elevated" if sc >= theta else ("watch" if sc >= 0.5 * theta else "normal")

    for rid, cfg in P.ROLES.items():
        k = cfg["k"]; payload = dict(common); payload["role"] = rid; payload["role_label"] = cfg["label"]; payload["granularity"] = cfg["granularity"]; payload["k"] = k
        series = {}
        for d in DISTRICTS:
            F = fused[d][0][idx]
            dd = dict(risk=[r1(x) for x in F])
            if cfg["env"]:
                dd["wbgt_max"] = [r1(x) for x in piv[d]["wbgt_max"][idx]]; dd["pm10"] = [r1(x) for x in piv[d]["pm10"][idx]]
            if cfg["can_see_counts"]:
                dd["streams"] = {}
                for s in cfg["streams"]:
                    vals = piv[d][s][idx]; shown, flags = P.suppress_series(vals, k)
                    dd["streams"][s] = dict(observed=shown, suppressed=[bool(f) for f in flags], expected=[r1(x) for x in mu[d][s][idx]])
            else:
                if cfg["granularity"] == "district-week":
                    wk_dates, _ = agg_weeks(piv[d]["incidents"], k)
                    dd["weekly"] = {}
                    for s in cfg["streams"]:
                        _, w = agg_weeks(piv[d][s], k); shown, flags = P.suppress_series(w, k)
                        dd["weekly"][s] = dict(observed=shown, suppressed=[bool(f) for f in flags])
                    dd["week_dates"] = wk_dates
                    dd["weekly_band"] = [risk_band(float(np.max(F[i:i + 7]))) for i in range(0, len(F), 7)][:len(wk_dates)]
                else:  # monthly
                    s_ = pd.Series(piv[d]["incidents"][idx], index=dts[idx]); mths = s_.resample("MS").sum().index
                    dd["month_dates"] = [m.strftime("%Y-%m") for m in mths]; dd["monthly"] = {}
                    for s in cfg["streams"]:
                        m = pd.Series(piv[d][s][idx], index=dts[idx]).resample("MS").sum().values; shown, flags = P.suppress_series(m, k)
                        dd["monthly"][s] = dict(observed=shown, suppressed=[bool(f) for f in flags])
                    Fm = pd.Series(F, index=dts[idx]).resample("MS").max().values
                    dd["monthly_band"] = [risk_band(float(x)) for x in Fm]
            series[d] = dd
        payload["series"] = series
        if cfg["actions"]:
            payload["alerts"] = alerts
        else:   # no operational detail, only where/when risk was elevated
            payload["alerts"] = [dict(id=a["id"], district=a["district"], open=a["open"], close=a["close"], open_i=a["open_i"], close_i=a["close_i"], tier=a["tier"], tiers=a["tiers"], hazard_label=a["hazard_label"]) for a in alerts]
        payload["backtest"] = BASE; payload["impact"] = impact; payload["robustness"] = json.load(open(BT.OUT / "robustness.json"))
        for s in list(payload["stream_names"]):
            if s not in cfg["streams"] and not cfg["can_see_counts"]: pass
        json.dump(payload, open(WEB / f"data_{rid}.json", "w"), separators=(",", ":"))
        print(rid, round((WEB / f"data_{rid}.json").stat().st_size / 1024), "KB")
    json.dump(dict(roles={r: {k_: v for k_, v in c.items()} for r, c in P.ROLES.items()}, k_default=P.K_DEFAULT, purpose=P.PURPOSE_STATEMENT), open(WEB / "roles.json", "w"), indent=1)
    tp = sum(1 for a in alerts if a["truth"]); hc = sum(1 for a in alerts if a["truth"] and a["truth"]["hazard_correct"])
    print("alerts", len(alerts), "TP", tp, "hazard correct", hc, "theta", theta, rho)
    print("impact", json.dumps(impact["total_excess"]), json.dumps(impact["after_prodrome"]), json.dumps(impact["after_weekly"]))


if __name__ == "__main__":
    main()
