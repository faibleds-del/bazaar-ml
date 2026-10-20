"""Score the latest Bazaar data with the trained model and list the best candidates.

    python signals.py
    python signals.py --top 40

This is a signal tool: read it, decide, and trade by hand in game.
"""
import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from features import add_features, load_mayor, load_snapshots, to_grid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--model-dir", default="models")
    ap.add_argument("--top", type=int, default=20)
    args = ap.parse_args()

    model_dir = Path(args.model_dir)
    meta = json.loads((model_dir / "meta.json").read_text())
    booster = lgb.Booster(model_file=str(model_dir / "model.txt"))

    raw = load_snapshots(args.data, days=3)  # enough history for the 24h features
    df = add_features(to_grid(raw, meta["freq"]), load_mayor(args.data),
                      meta["freq"], meta["horizon"], meta["tax"])
    latest = df[df["time"] == df["time"].max()]
    latest = latest[latest["log_coin_vol"] >= np.log1p(meta["min_coin_vol"])].copy()

    age = pd.Timestamp.now(tz="UTC") - pd.to_datetime(raw["ts"].max(), unit="ms", utc=True)
    print(f"Latest snapshot is {age.total_seconds() / 60:.0f} min old. "
          f"Predicting the next {meta['horizon']:g}h.\n")

    latest["pred_move"] = np.expm1(booster.predict(latest[meta["features"]]))
    latest["edge"] = latest["pred_move"] - latest["spread_pct"] - meta["tax"]
    cols = {"item": "item", "bid": "insta-sell", "ask": "insta-buy", "spread_pct": "spread",
            "pred_move": "pred move", "edge": "edge after costs"}
    table = latest.sort_values("edge", ascending=False).head(args.top)[list(cols)].rename(columns=cols)
    pct = lambda v: f"{v:+.2%}"
    print(table.to_string(index=False, formatters={
        "insta-sell": "{:,.1f}".format, "insta-buy": "{:,.1f}".format,
        "spread": pct, "pred move": pct, "edge after costs": pct}))
    print("\nPositive edge = model expects the move to beat spread + tax. "
          "Check the test-set backtest in train.py output before trusting any of this.")


if __name__ == "__main__":
    main()
