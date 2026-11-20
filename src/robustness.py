"""Single-feed artefact test: inject a 5-day x3 spike (duplicated batch / pipeline glitch) into ONE feed of ONE district
with no real event present, and count how many such glitches raise an alert, with and without the corroboration gate."""
import pickle, json, copy
import numpy as np, pandas as pd
import backtest as BT
from simulate import STREAMS, DISTRICTS
from detect import cusum, fuse, alerts_from_fused

def main():
    res = pickle.load(open(BT.OUT / "scores.pkl", "rb"))
    ev = pd.read_csv(BT.DATA / "events_truth.csv", parse_dates=["onset"])
    (theta, rho), *_ = BT.run_variant(res, "full", ev)
    out = dict(theta=theta, rho=rho, trials=120,
               gate_rate=BT.glitch_rate(res, ev, theta, rho, BT.TEST, n_trials=120, gate_on=True),
               no_gate_rate=BT.glitch_rate(res, ev, theta, rho, BT.TEST, n_trials=120, gate_on=False),
               design_max=BT.GLITCH_MAX)
    json.dump(out, open(BT.OUT / "robustness.json", "w"), indent=1); print(out)

if __name__ == "__main__":
    main()
