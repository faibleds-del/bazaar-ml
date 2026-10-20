"""Grab one Bazaar snapshot and save it under data/snapshots/.

Runs from GitHub Actions every ~5 minutes. Standard library only, so the
workflow doesn't need a pip install step.

Column naming (Hypixel's own names are confusing, so we rename):
  ask  = lowest sell offer  = what you pay to "Buy Instantly"   (buy_summary[0])
  bid  = highest buy order  = what you get to "Sell Instantly"  (sell_summary[0])
"""
import csv
import gzip
import io
import json
import os
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

BAZAAR_URL = "https://api.hypixel.net/v2/skyblock/bazaar"
ELECTION_URL = "https://api.hypixel.net/v2/resources/skyblock/election"
DATA = Path(os.environ.get("DATA_DIR", "data"))
TOP_N = int(os.environ.get("TOP_N", "500"))  # keep the N items with the most coin volume

FIELDS = [
    "ts", "item",
    "ask", "bid",                # top of book
    "ask_qs", "bid_qs",          # Hypixel's quick_status prices (avg of top 2% of book)
    "ask_vol", "bid_vol",        # units sitting in sell offers / buy orders
    "ask_orders", "bid_orders",  # number of sell offers / buy orders
    "buy_week", "sell_week",     # units instant-bought / instant-sold in the last 7 days
    "ask_depth5", "bid_depth5",  # units in the best 5 price levels
]


def fetch(url, tries=3):
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "bazaar-ml-collector"})
            with urllib.request.urlopen(req, timeout=20) as r:
                data = json.load(r)
            if not data.get("success"):
                raise RuntimeError(f"API returned success=false: {data.get('cause')}")
            return data
        except Exception as e:
            if i == tries - 1:
                raise
            print(f"fetch {url} failed ({e}), retrying", file=sys.stderr)
            time.sleep(5 * (i + 1))


def best(levels):
    return levels[0]["pricePerUnit"] if levels else None


def depth(levels, n=5):
    return sum(level["amount"] for level in levels[:n])


def build_rows(bazaar):
    ts = bazaar["lastUpdated"]
    ranked = []
    for item, p in bazaar["products"].items():
        q = p["quick_status"]
        sells, buys = p.get("buy_summary", []), p.get("sell_summary", [])
        ask, bid = best(sells), best(buys)
        if ask is None or bid is None:
            continue  # one-sided book, nothing to model
        coin_volume = q["buyMovingWeek"] * ask + q["sellMovingWeek"] * bid
        ranked.append((coin_volume, [
            ts, item,
            ask, bid,
            round(q["buyPrice"], 2), round(q["sellPrice"], 2),
            q["buyVolume"], q["sellVolume"],
            q["buyOrders"], q["sellOrders"],
            q["buyMovingWeek"], q["sellMovingWeek"],
            depth(sells), depth(buys),
        ]))
    ranked.sort(key=lambda x: x[0], reverse=True)
    return [row for _, row in ranked[:TOP_N]]


def save_snapshot(ts, rows):
    t = datetime.fromtimestamp(ts / 1000, tz=timezone.utc)
    path = DATA / "snapshots" / t.strftime("%Y/%m/%d") / f"{t:%H%M%S}.csv.gz"
    if path.exists():
        print(f"{path} already saved, API hasn't updated")
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(FIELDS)
    writer.writerows(rows)
    with open(path, "wb") as f, gzip.GzipFile(fileobj=f, mode="wb", mtime=0) as gz:
        gz.write(buf.getvalue().encode())
    return path


def save_mayor(ts):
    """Append mayor, minister and active perks to data/mayor.csv, only when they change.

    Perks decide events like Fishing Festival (Marina) and Mining Fiesta (Cole),
    and Mayor Jerry swaps perks mid-term, so perks are tracked, not just names.
    """
    try:
        mayor = fetch(ELECTION_URL, tries=2).get("mayor") or {}
    except Exception as e:
        print(f"election fetch failed, skipping mayor: {e}", file=sys.stderr)
        return
    minister = mayor.get("minister") or {}
    perks = [p.get("name", "") for p in mayor.get("perks") or []]
    if (minister.get("perk") or {}).get("name"):
        perks.append(minister["perk"]["name"])
    current = [mayor.get("name", ""), minister.get("name", ""), "|".join(perks)]
    path = DATA / "mayor.csv"
    if path.exists():
        with open(path, newline="") as f:
            rows = list(csv.reader(f))
        if len(rows) > 1 and rows[-1][1:] == current:
            return
    is_new = not path.exists()
    with open(path, "a", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        if is_new:
            writer.writerow(["ts", "mayor", "minister", "perks"])
        writer.writerow([ts, *current])
    print(f"mayor changed: {current}")


def main():
    bazaar = fetch(BAZAAR_URL)
    rows = build_rows(bazaar)
    path = save_snapshot(bazaar["lastUpdated"], rows)
    if path:
        print(f"saved {len(rows)} items to {path} ({path.stat().st_size / 1024:.1f} KB)")
    save_mayor(bazaar["lastUpdated"])


if __name__ == "__main__":
    main()
