"""Privacy-by-design controls for Prodrome.

1. Aggregate-only schema (guard raises if a person-level column ever appears).
2. Small-cell suppression: any displayed count below K is replaced by a '<K' flag; complementary
   suppression is applied when a total would let a suppressed cell be back-calculated.
3. Role-based views: each role sees only the fields and granularity needed for its function.
4. Purpose limitation: outputs are tagged and the UI states they are not for individual employment,
   disciplinary, immigration or insurance decisions.
"""
import numpy as np

K_DEFAULT = 5
FORBIDDEN_COLUMNS = {"name", "employee_id", "person_id", "emirates_id", "passport", "dob", "phone", "email",
                     "employer_name", "worker_id", "nationality", "address", "lat", "lon", "gps"}

PURPOSE_STATEMENT = ("Aggregate population-health surveillance only. Outputs must not be used for individual "
                     "employment, disciplinary, immigration, visa or insurance decisions, nor to identify "
                     "individual workers or employers as sources of illness.")

ROLES = {
    "public_health": dict(label="Public-health surveillance officer", granularity="district-day", k=5,
                          streams=["heat_visits", "gi_visits", "resp_visits", "lab_gi_pos", "absence", "incidents"],
                          env=True, actions=True, can_see_counts=True),
    "osh_inspector": dict(label="OSH / labour inspectorate", granularity="district-day", k=5,
                          streams=["heat_visits", "resp_visits", "absence", "incidents"],
                          env=True, actions=True, can_see_counts=True),
    "employer_hse": dict(label="Employer HSE lead (own sector aggregate)", granularity="district-week", k=10,
                         streams=["heat_visits", "incidents", "absence"],
                         env=True, actions=False, can_see_counts=False),
    "policy_analyst": dict(label="Policy / regulatory analyst", granularity="district-month", k=10,
                           streams=["heat_visits", "gi_visits", "resp_visits", "absence", "incidents"],
                           env=True, actions=False, can_see_counts=False),
}


def guard_schema(columns):
    bad = {c.lower() for c in columns} & FORBIDDEN_COLUMNS
    if bad:
        raise ValueError(f"Individual-level fields are not permitted in Prodrome: {sorted(bad)}")
    return True


def suppress(value, k=K_DEFAULT):
    """Return (display_value, suppressed_flag). Counts in 1..k-1 are suppressed; zeros are shown."""
    v = int(round(float(value)))
    if 0 < v < k:
        return None, True
    return v, False


def suppress_series(vals, k=K_DEFAULT):
    out, flags = [], []
    for v in vals:
        d, f = suppress(v, k)
        out.append(d); flags.append(f)
    return out, flags


def complementary(cells, k=K_DEFAULT):
    """If exactly one cell in a group is suppressed, also suppress the smallest other cell so the
    suppressed value cannot be recovered from a published total."""
    disp = [suppress(c, k) for c in cells]
    flags = [f for _, f in disp]
    if sum(flags) == 1:
        cand = [i for i, (v, f) in enumerate(disp) if not f and v is not None and v > 0]
        if cand:
            j = min(cand, key=lambda i: disp[i][0]); disp[j] = (None, True)
    return disp
