"""Download one year of half-hourly GB electricity market data from the
Elexon Insights API (free, no API key) and save it as data/gb_power_2025.csv.

Columns (MW unless stated, one row per half-hour settlement period):
  price      day-ahead/within-day market index price, GBP/MWh (APX MID)
  demand     initial transmission system demand outturn (ITSDO)
  wind       metered wind generation (FUELHH)
  solar      solar generation estimate (B1630)
  ccgt, ocgt, nuclear, biomass, pumped_storage, interconnectors (net imports)

Usage:  python3 get_data.py            (takes a few minutes)
"""
import json, time, urllib.request
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

BASE = "https://data.elexon.co.uk/bmrs/api/v1"
START, END = datetime(2025, 1, 1, tzinfo=timezone.utc), datetime(2026, 1, 1, tzinfo=timezone.utc)
OUT = Path(__file__).parent / "data" / "gb_power_2025.csv"


def get(url):
    """Fetch a JSON endpoint and return its 'data' list, retrying on errors."""
    for attempt in range(4):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return json.load(r).get("data", [])
        except Exception as e:  # network hiccup or rate limit
            if attempt == 3:
                raise
            time.sleep(2 * (attempt + 1))


def iso(t):
    return t.strftime("%Y-%m-%dT%H:%MZ")


rows = {}


def row(d, p):
    return rows.setdefault((d, p), {"date": d, "period": p})


t = START
while t < END:  # the API allows at most 7 days per request
    t2 = min(t + timedelta(days=7), END)
    d1, d2 = t.date().isoformat(), (t2 - timedelta(days=1)).date().isoformat()
    print(f"{d1} to {d2}")

    for m in get(f"{BASE}/balancing/pricing/market-index?from={iso(t)}&to={iso(t2)}&dataProviders=APXMIDP&format=json"):
        row(m["settlementDate"], m["settlementPeriod"])["price"] = m["price"]

    for m in get(f"{BASE}/demand/outturn?settlementDateFrom={d1}&settlementDateTo={d2}&format=json"):
        row(m["settlementDate"], m["settlementPeriod"])["demand"] = m["initialTransmissionSystemDemandOutturn"]

    fuel_map = {"WIND": "wind", "CCGT": "ccgt", "OCGT": "ocgt", "NUCLEAR": "nuclear",
                "BIOMASS": "biomass", "PS": "pumped_storage"}
    for m in get(f"{BASE}/datasets/FUELHH?settlementDateFrom={d1}&settlementDateTo={d2}&format=json"):
        r = row(m["settlementDate"], m["settlementPeriod"])
        f = m["fuelType"]
        if f.startswith("INT"):
            r["interconnectors"] = r.get("interconnectors", 0) + m["generation"]
        elif f in fuel_map:
            r[fuel_map[f]] = m["generation"]

    for m in get(f"{BASE}/generation/actual/per-type/wind-and-solar?from={iso(t)}&to={iso(t2)}&format=json"):
        if m["psrType"] == "Solar":
            row(m["settlementDate"], m["settlementPeriod"])["solar"] = m["quantity"]

    t = t2

df = pd.DataFrame(rows.values()).sort_values(["date", "period"])
df = df[(df["date"] >= "2025-01-01") & (df["date"] <= "2025-12-31")]
cols = ["date", "period", "price", "demand", "wind", "solar", "ccgt", "ocgt",
        "nuclear", "biomass", "pumped_storage", "interconnectors"]
OUT.parent.mkdir(exist_ok=True)
df[cols].to_csv(OUT, index=False)
print(f"Saved {len(df)} half-hours to {OUT}")
