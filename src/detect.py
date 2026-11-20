"""Prodrome detection engine.

Pipeline (all causal - day t only uses data that would have been *available* on day t):
 1. Per district x stream weather-adjusted baseline: quasi-Poisson GLM with day-of-week, annual harmonics and
    environmental covariates (WBGT heat dose, PM10), refit every 14 days on the trailing 365 days with a
    robust second pass that drops previously-flagged days (so past outbreaks do not poison the baseline).
 2. Standardised residual z -> one-sided CUSUM per stream (persistent shift, not one noisy day).
 3. Evidence fusion: an alert needs corroboration from >=2 streams spanning >=2 signal families
    (clinical / occupational) - this is what suppresses single-feed noise and alert fatigue.
 4. Explainability: per-stream contribution share, % of excess explained by weather vs unexplained, hazard
    signature match -> proportionate action + responsible role.
"""
import warnings
import numpy as np, pandas as pd
import statsmodels.api as sm
from simulate import STREAMS, FAMILY, LATENCY

warnings.filterwarnings("ignore")
REFIT_EVERY, WINDOW, MIN_HIST = 14, 365, 300
CUSUM_K = 0.5


def design(df, use_env=True):
    """df: index=date, columns wbgt_max, pm10. Returns feature matrix."""
    idx = df.index
    X = pd.DataFrame(index=idx)
    X["const"] = 1.0
    dow = idx.dayofweek.values
    for k in range(1, 7):
        X[f"dow{k}"] = (dow == k).astype(float)
    doy = idx.dayofyear.values
    for h in (1, 2):
        X[f"s{h}"] = np.sin(2 * np.pi * h * doy / 365.25)
        X[f"c{h}"] = np.cos(2 * np.pi * h * doy / 365.25)
    if use_env:
        X["heat"] = np.maximum(df.wbgt_max.values - 27.0, 0) ** 1.4
        X["pm"] = np.maximum(df.pm10.values - 40, 0) / 50.0
    return X


ENV_COLS = ["heat", "pm"]


def fit_stream(y, X, mask_ok):
    """Robust two-pass quasi-Poisson fit. Returns (params, phi) or None."""
    keep = mask_ok.copy()
    beta, phi = None, 1.0
    for _ in range(2):
        if keep.sum() < MIN_HIST * 0.8:
            return None
        try:
            m = sm.GLM(y[keep], X.values[keep], family=sm.families.Poisson()).fit(maxiter=60)
        except Exception:
            return None
        mu = np.exp(np.clip(X.values @ m.params, -20, 20))
        pr = (y - mu) / np.sqrt(mu + 0.3)
        phi = max(1.0, float(np.sum(pr[keep] ** 2) / max(keep.sum() - X.shape[1], 1)))
        beta = m.params
        keep = mask_ok & (pr < 2.5 * np.sqrt(phi))  # drop upper-tail residual days (past events) for pass 2
    return beta, phi


def residual_streams(long, district, use_env=True, drop_env_cols=False, missing_mask=None):
    """Compute causal expected counts & z for every stream of one district over the whole period.
    missing_mask: dict stream -> boolean array (True = data day unavailable)."""
    sub = long[long.district == district]
    env = sub[sub.stream.isin(["wbgt_max", "pm10"])].pivot(index="date", columns="stream", values="count")
    env.index = pd.to_datetime(env.index)
    X = design(env, use_env=use_env and not drop_env_cols)
    dates = X.index
    n = len(dates)
    out = {}
    for s in STREAMS:
        y = sub[sub.stream == s].set_index(pd.to_datetime(sub[sub.stream == s].date))["count"].reindex(dates).values.astype(float)
        mu = np.full(n, np.nan); phi_t = np.full(n, np.nan)
        avail = np.ones(n, bool) if missing_mask is None else ~missing_mask.get(s, np.zeros(n, bool))
        for t0 in range(MIN_HIST, n, REFIT_EVERY):
            lo = max(0, t0 - WINDOW)
            ok = np.zeros(n, bool); ok[lo:t0] = True; ok &= avail
            r = fit_stream(y, X, ok)
            if r is None:
                continue
            beta, phi = r
            sl = slice(t0, min(t0 + REFIT_EVERY, n))
            mu[sl] = np.exp(np.clip(X.values[sl] @ beta, -20, 20)); phi_t[sl] = phi
            # counterfactual: environment at climatological "neutral" level (for weather-explained share)
        out[s] = dict(y=y, mu=mu, phi=phi_t, avail=avail)
    return dates, X, out


def neutral_mu(X, beta_cache=None):
    return None


DELTA = 0.8   # design shift: CUSUM is tuned to detect an ~80% rise over the weather-adjusted baseline


def cusum(y, mu, phi, avail, delta=DELTA, cap=10.0, leak=0.9):
    """Poisson log-likelihood-ratio CUSUM (nats of evidence for 'rate x (1+delta)' vs baseline), overdispersion-scaled.
    Log-likelihoods add across independent feeds, so the fused score is a principled combined evidence."""
    s = np.zeros(len(y)); acc = 0.0; lg = np.log1p(delta)
    for i in range(len(y)):
        if np.isnan(mu[i]) or not avail[i]:
            acc = acc * 0.9  # decay through gaps rather than reset / fabricate evidence
        else:
            llr = (y[i] * lg - mu[i] * delta) / phi[i]
            acc = min(max(0.0, acc * leak + llr), cap)
        s[i] = acc
    return s


def stream_scores(dates, out):
    res = {}
    for s, o in out.items():
        z = (o["y"] - o["mu"]) / np.sqrt(o["phi"] * o["mu"] + 0.5)
        res[s] = dict(z=z, cusum=cusum(o["y"], o["mu"], o["phi"], o["avail"]), **o)
    return res


STREAM_W = {"heat_visits": 1.0, "gi_visits": 1.0, "resp_visits": 1.0, "lab_gi_pos": 0.8, "absence": 0.8, "incidents": 0.9}
ACTIVE_TAU = 1.6          # a stream counts as "elevated" for corroboration when its CUSUM exceeds this


def fuse(scores, theta, streams=None, latency=None, n=None, rho=0.25):
    """Fused evidence (sum of per-stream LLR-CUSUMs). Latency shifts each stream so day t only sees data up to t-lag.
    A stream is 'active' when its evidence > rho*theta. Gate: >=2 active streams spanning >=2 distinct families
    (service / laboratory / workplace) - corroboration across materially different signal types."""
    streams = streams or list(STREAMS)
    latency = LATENCY if latency is None else latency
    F = np.zeros(n); active = np.zeros(n, int)
    fams = {}
    contrib = {}
    for s in streams:
        c = scores[s]["cusum"]
        lag = latency.get(s, 0)
        c = np.concatenate([np.zeros(lag), c[: n - lag]]) if lag else c
        contrib[s] = STREAM_W[s] * np.minimum(c, 10.0)
        F += contrib[s]
        a = (c > rho * theta)
        active += a.astype(int)
        fams.setdefault(FAMILY[s], np.zeros(n, bool))
        fams[FAMILY[s]] |= a
    n_fam = sum(f.astype(int) for f in fams.values())
    gate = (active >= 2) & (n_fam >= 2)
    return F, active, gate, contrib


def alerts_from_fused(dates, F, gate, theta, close_ratio=0.5, cooldown=5):
    """Open alert when F>theta & gate. Close after F<theta*close_ratio for 3 days. Return list of (open_idx, close_idx)."""
    out = []; i = 0; n = len(F)
    while i < n:
        if F[i] > theta and gate[i]:
            j = i
            low = 0
            while j < n:
                low = low + 1 if F[j] < theta * close_ratio else 0
                if low >= 3: break
                j += 1
            out.append((i, min(j, n - 1)))
            i = j + cooldown
        else:
            i += 1
    return out
