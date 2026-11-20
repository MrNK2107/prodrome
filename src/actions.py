"""Hazard signature matching + proportionate action recommendations with responsible roles.
Roles are generic functions (not named agencies). Every recommendation carries evidence + assumptions.
Hard boundary: outputs are aggregate, district-level, and never usable for individual or disciplinary decisions."""
import numpy as np

PROTOTYPES = {
    "heat_stress":  {"heat_visits": 1.0, "incidents": 0.5, "absence": 0.3},
    "gi_outbreak":  {"gi_visits": 1.0, "lab_gi_pos": 1.1, "absence": 0.4},
    "chemical_exposure": {"incidents": 1.0, "resp_visits": 0.8, "absence": 0.3},
    "respiratory_cluster": {"resp_visits": 1.0, "absence": 0.5},
}
LABEL = {"heat_stress": "Heat-stress cluster", "gi_outbreak": "Gastro-intestinal outbreak",
         "chemical_exposure": "Possible occupational chemical exposure", "respiratory_cluster": "Respiratory illness cluster"}
TRUTH_MAP = {"heat_cluster": "heat_stress", "gi_outbreak": "gi_outbreak", "chemical_exposure": "chemical_exposure", "resp_cluster": "respiratory_cluster"}

PLAYBOOK = {
    "heat_stress": {
        "Watch":    ("Issue heat advisory to sector HSE leads; confirm midday-work-break compliance.", "Occupational health & safety inspectorate"),
        "Advisory": ("Targeted site inspections of shade/water/rest provision at construction & outdoor sites in the district.", "Labour / OSH inspectorate + employer HSE leads"),
        "Action":   ("Mandate temporary work-hour restriction for outdoor sites in district; deploy cooling/hydration stations; clinic surge-readiness.", "Regulator (OSH) with public-health officer"),
    },
    "gi_outbreak": {
        "Watch":    ("Ask lab and clinics for stool-culture prioritisation; no field action yet.", "Public-health surveillance officer"),
        "Advisory": ("Environmental-health visit to catering/accommodation facilities serving the affected workforce.", "Environmental health / food-safety officer"),
        "Action":   ("Outbreak investigation: water & food source sampling, isolate affected accommodation blocks, hygiene intervention.", "Public-health emergency team + accommodation operators"),
    },
    "chemical_exposure": {
        "Watch":    ("Cross-check shared incident logs for common process/location; no individual follow-up from this system.", "Industrial-zone OSH coordinator"),
        "Advisory": ("Request process-safety and ventilation review from facilities in the district; inform occupational physicians.", "OSH inspectorate + facility safety managers"),
        "Action":   ("Rapid joint inspection; air monitoring at implicated facilities; precautionary controls pending findings.", "Regulator (OSH/environment) + occupational-health physician"),
    },
    "respiratory_cluster": {
        "Watch":    ("Increase sentinel testing; review ventilation/crowding advisories.", "Public-health surveillance officer"),
        "Advisory": ("Targeted infection-prevention messaging and ventilation checks at affected workplaces and accommodation.", "Public-health officer + employer HSE leads"),
        "Action":   ("Cluster investigation, targeted testing/vaccination catch-up, temporary shift-density measures.", "Public-health emergency team"),
    },
}


def classify(contrib, wbgt_hi):
    """contrib: dict stream->contribution. Cosine similarity to prototypes; heat gets a boost when WBGT is high."""
    keys = ["heat_visits", "gi_visits", "resp_visits", "lab_gi_pos", "absence", "incidents"]
    v = np.array([contrib.get(k, 0.0) for k in keys]); v = v / (np.linalg.norm(v) + 1e-9)
    best, sims = None, {}
    for h, proto in PROTOTYPES.items():
        p = np.array([proto.get(k, 0.0) for k in keys]); p = p / np.linalg.norm(p)
        sim = float(v @ p)
        if h == "heat_stress": sim += 0.12 if wbgt_hi else -0.10
        if h == "chemical_exposure": sim += -0.10 if wbgt_hi else 0.03
        sims[h] = sim
    best = max(sims, key=sims.get)
    srt = sorted(sims.values(), reverse=True)
    return best, sims, float(srt[0] - srt[1])


def tier(F, theta, n_active):
    if F >= 2.2 * theta and n_active >= 3: return "Action"
    if F >= 1.4 * theta: return "Advisory"
    return "Watch"


def recommend(hazard, t):
    act, role = PLAYBOOK[hazard][t]
    return act, role
