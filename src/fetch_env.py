"""Fetch REAL environmental data (Open-Meteo, CC BY 4.0) for three Abu Dhabi emirate zones.
Daily work-hours WBGT proxy + PM10/PM2.5. Cached to data/env_<zone>.csv."""
import json, sys, urllib.request, pathlib
import numpy as np, pandas as pd

ZONES = {"abudhabi": (24.35, 54.50), "alain": (24.21, 55.76), "ruwais": (24.11, 52.73)}
START, END = "2022-01-01", "2025-12-31"
DATA = pathlib.Path(__file__).resolve().parent.parent / "data"

def get(url):
    with urllib.request.urlopen(url, timeout=60) as r:
        return json.load(r)

def wet_bulb(T, RH):  # Stull (2011)
    return (T*np.arctan(0.151977*np.sqrt(RH+8.313659)) + np.arctan(T+RH)
            - np.arctan(RH-1.676331) + 0.00391838*RH**1.5*np.arctan(0.023101*RH) - 4.686035)

def fetch(zone, lat, lon):
    w = get(f"https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}"
            f"&start_date={START}&end_date={END}&hourly=temperature_2m,relative_humidity_2m,shortwave_radiation"
            f"&timezone=Asia%2FDubai")
    h = pd.DataFrame(w["hourly"]); h["time"] = pd.to_datetime(h["time"])
    Tw = wet_bulb(h.temperature_2m, h.relative_humidity_2m)
    Tg = h.temperature_2m + 0.015*h.shortwave_radiation          # crude globe temp
    h["wbgt"] = 0.7*Tw + 0.2*Tg + 0.1*h.temperature_2m           # outdoor WBGT proxy
    h["date"] = h.time.dt.date; h["hr"] = h.time.dt.hour
    work = h[(h.hr >= 10) & (h.hr <= 16)]
    d = work.groupby("date").agg(wbgt_max=("wbgt", "max"), tmax=("temperature_2m", "max"), rh_mean=("relative_humidity_2m", "mean"))
    a = get(f"https://air-quality-api.open-meteo.com/v1/air-quality?latitude={lat}&longitude={lon}"
            f"&start_date={START}&end_date={END}&hourly=pm10,pm2_5&timezone=Asia%2FDubai")
    q = pd.DataFrame(a["hourly"]); q["time"] = pd.to_datetime(q["time"]); q["date"] = q.time.dt.date
    qd = q.groupby("date").agg(pm10=("pm10", "mean"), pm25=("pm2_5", "mean"))
    out = d.join(qd, how="left"); out.index = pd.to_datetime(out.index); out.index.name = "date"
    out[["pm10", "pm25"]] = out[["pm10", "pm25"]].interpolate(limit=5).bfill().ffill()
    return out

if __name__ == "__main__":
    for z, (la, lo) in ZONES.items():
        f = DATA / f"env_{z}.csv"
        if f.exists() and "--force" not in sys.argv: print("cached", f.name); continue
        df = fetch(z, la, lo); df.to_csv(f); print(z, df.shape, df.wbgt_max.describe()[["min","mean","max"]].round(1).to_dict(), "pm10max", round(df.pm10.max(),1))
