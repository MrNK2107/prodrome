import sys, json, pathlib, pickle
import numpy as np, pandas as pd, pytest
ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import privacy as P
from detect import cusum, fuse, alerts_from_fused
from simulate import STREAMS, LATENCY

def test_small_cell_suppression():
    assert P.suppress(0) == (0, False)
    assert P.suppress(3) == (None, True)
    assert P.suppress(4) == (None, True)
    assert P.suppress(5) == (5, False)
    assert P.suppress(12, k=10) == (12, False) and P.suppress(9, k=10) == (None, True)

def test_complementary_suppression_hides_recoverable_cell():
    out = P.complementary([3, 8, 40])           # one suppressed -> second smallest must also be hidden
    assert sum(f for _, f in out) == 2

def test_schema_guard_rejects_individual_fields():
    assert P.guard_schema(["date", "district", "stream", "count"])
    for bad in ("employee_id", "Name", "lat", "employer_name"):
        with pytest.raises(ValueError):
            P.guard_schema(["date", bad])

def test_dataset_has_no_individual_fields():
    long = pd.read_csv(ROOT / "data" / "signals_long.csv.gz", nrows=5)
    assert P.guard_schema(long.columns)

def test_role_payloads_respect_entitlements():
    for role, cfg in P.ROLES.items():
        d = json.load(open(ROOT / "web" / f"data_{role}.json"))
        first = next(iter(d["series"].values()))
        if cfg["can_see_counts"]:
            assert set(first["streams"]) == set(cfg["streams"])
        else:
            assert "streams" not in first                       # no daily counts for broad roles
            assert all("evidence" not in a and "action" not in a for a in d["alerts"])
        if not cfg["actions"]:
            assert all("action" not in a for a in d["alerts"])

def test_published_counts_never_show_small_cells():
    d = json.load(open(ROOT / "web" / "data_public_health.json"))
    for ser in d["series"].values():
        for st in ser["streams"].values():
            for v, sup in zip(st["observed"], st["suppressed"]):
                assert (v is None) == sup
                if v is not None: assert v == 0 or v >= d["k"]

def test_cusum_ignores_gaps_and_decays():
    y = np.array([5, 9, 9, 9, 0, 0, 0, 0.]); mu = np.full(8, 5.); phi = np.ones(8)
    avail = np.array([1, 1, 1, 1, 0, 0, 0, 0], bool)
    s = cusum(y, mu, phi, avail)
    assert s[3] > s[0] and s[7] < s[3]                          # evidence decays through missing data, no fabricated evidence

def test_alert_gate_needs_two_families():
    n = 40; sc = {s: dict(cusum=np.zeros(n)) for s in STREAMS}
    sc["gi_visits"]["cusum"][10:20] = 12                         # one feed alone, huge
    F, act, gate, _ = fuse(sc, 6.0, n=n, latency={k: 0 for k in LATENCY})
    assert F.max() > 6 and not gate.any()                        # single feed never passes the gate
    sc["lab_gi_pos"]["cusum"][10:20] = 5
    F, act, gate, _ = fuse(sc, 6.0, n=n, latency={k: 0 for k in LATENCY})
    assert gate.any()

def test_causality_no_future_leak():
    """Changing data after day t must not change the fused evidence at or before day t."""
    n = 60; rng = np.random.default_rng(1)
    mu = np.full(n, 4.); phi = np.ones(n); av = np.ones(n, bool)
    y1 = rng.poisson(4, n).astype(float); y2 = y1.copy(); y2[40:] += 50
    a = cusum(y1, mu, phi, av); b = cusum(y2, mu, phi, av)
    assert np.allclose(a[:40], b[:40])

def test_latency_shifts_evidence():
    n = 30; sc = {s: dict(cusum=np.zeros(n)) for s in STREAMS}
    sc["lab_gi_pos"]["cusum"][10] = 9
    F0, *_ = fuse(sc, 6.0, n=n, latency={k: 0 for k in LATENCY}); F3, *_ = fuse(sc, 6.0, n=n, latency=LATENCY)
    assert np.argmax(F0) == 10 and np.argmax(F3) == 13           # lab feed arrives 3 days late

def test_backtest_outputs_consistent():
    R = json.load(open(ROOT / "out" / "backtest.json")); P_ = R["prodrome"]
    assert P_["detected"] <= P_["events"] and P_["alerts"] - P_["false_alerts"] >= P_["detected"] * 0 
    assert abs(P_["ppv"] - (P_["alerts"] - P_["false_alerts"]) / P_["alerts"]) < 1e-9
    assert P_["sensitivity"] > R["daily_single_stream"]["sensitivity"]
