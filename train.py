"""Train a LightGBM model on collected Bazaar data and evaluate it honestly.

    python train.py                  # predict the 1h move
    python train.py --horizon 6      # predict the 6h move

Evaluation is strictly time-based: train on the oldest data, tune on the
next chunk, test on the newest chunk, with a gap between them so targets
never leak across the boundary.
"""
import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from features import TAX, add_features, feature_columns, load_mayor, load_snapshots, to_grid

PARAMS = dict(
    learning_rate=0.03, num_leaves=31, min_child_samples=200,
    subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
    reg_lambda=1.0, verbose=-1,
)


def rank_corr(a, b):
    return pd.Series(np.asarray(a)).rank().corr(pd.Series(np.asarray(b)).rank())


def split_by_time(df, valid_frac, test_frac, gap):
    times = pd.Series(df["time"].unique()).sort_values().reset_index(drop=True)
    t_valid = times.iloc[int(len(times) * (1 - valid_frac - test_frac))]
    t_test = times.iloc[int(len(times) * (1 - test_frac))]
    train = df[df["time"] < t_valid - gap]
    valid = df[(df["time"] >= t_valid) & (df["time"] < t_test - gap)]
    test = df[df["time"] >= t_test]
    return train, valid, test


def non_overlapping(trades, horizon):
    """One position per item at a time, so a single move isn't counted 6 times."""
    keep = []
    for _, g in trades.sort_values("time").groupby("item"):
        last = None
        for idx, t in zip(g.index, g["time"]):
            if last is None or t >= last + horizon:
                keep.append(idx)
                last = t
    return trades.loc[keep]


def describe(name, trades):
    if trades.empty:
        return f"  {name:<28} no trades"
    r = trades["trade_ret"]
    return (f"  {name:<28} {len(r):>7} trades  mean {r.mean():+7.2%}  "
            f"median {r.median():+7.2%}  win rate {(r > 0).mean():6.1%}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--freq", default="10min", help="time grid size")
    ap.add_argument("--horizon", type=float, default=1.0, help="hours ahead to predict")
    ap.add_argument("--tax", type=float, default=TAX)
    ap.add_argument("--min-coin-vol", type=float, default=50e6,
                    help="skip items trading less than this many coins per week")
    ap.add_argument("--valid-frac", type=float, default=0.15)
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--edge", type=float, default=0.0,
                    help="extra predicted return required beyond spread + tax to trade")
    ap.add_argument("--out", default="models")
    args = ap.parse_args()

    raw = load_snapshots(args.data)
    df = add_features(to_grid(raw, args.freq), load_mayor(args.data), args.freq, args.horizon, args.tax)
    df = df.dropna(subset=["y", "trade_ret"])
    df = df[df["log_coin_vol"] >= np.log1p(args.min_coin_vol)]
    feats = feature_columns(df)

    days = (df["time"].max() - df["time"].min()) / pd.Timedelta("1D")
    print(f"{len(df):,} rows, {df['item'].nunique()} items, {days:.1f} days of data")
    if days < 7:
        print("WARNING: under a week of data. Results will be noisy; treat them as a smoke test.")

    gap = pd.Timedelta(hours=args.horizon)
    train, valid, test = split_by_time(df, args.valid_frac, args.test_frac, gap)
    for name, part in (("train", train), ("valid", valid), ("test", test)):
        print(f"  {name:<5} {part['time'].min():%Y-%m-%d %H:%M} -> {part['time'].max():%Y-%m-%d %H:%M}  ({len(part):,} rows)")

    # Clip target outliers (manipulation spikes) for training only
    lo, hi = train["y"].quantile([0.005, 0.995])
    model = lgb.LGBMRegressor(n_estimators=3000, **PARAMS)
    model.fit(
        train[feats], train["y"].clip(lo, hi),
        eval_set=[(valid[feats], valid["y"].clip(lo, hi))],
        callbacks=[lgb.early_stopping(100, verbose=False)],
    )
    pred = model.predict(test[feats])
    y = test["y"].clip(lo, hi).to_numpy()

    # Prediction quality vs. the "price stays the same" baseline
    rmse_model = np.sqrt(np.mean((pred - y) ** 2))
    rmse_base = np.sqrt(np.mean(y ** 2))
    moved = np.abs(y) > 1e-4
    per_time_ic = (test.assign(p=pred).groupby("time")
                   .apply(lambda d: rank_corr(d["p"], d["y"]), include_groups=False))
    metrics = {
        "best_iteration": int(model.best_iteration_),
        "rmse_model": float(rmse_model),
        "rmse_baseline": float(rmse_base),
        "rmse_improvement": float(1 - rmse_model / rmse_base),
        "direction_accuracy": float(np.mean(np.sign(pred[moved]) == np.sign(y[moved]))),
        "rank_ic": float(rank_corr(pred, y)),
        "mean_cross_sectional_ic": float(per_time_ic.mean()),
    }
    print(f"\nTest set ({args.horizon:g}h horizon), {metrics['best_iteration']} trees")
    print(f"  RMSE   model {rmse_model:.5f}  vs  no-change baseline {rmse_base:.5f}  "
          f"({metrics['rmse_improvement']:+.2%})")
    print(f"  direction accuracy  {metrics['direction_accuracy']:.1%}  (50% = coin flip)")
    print(f"  rank IC             {metrics['rank_ic']:+.3f}  (0 = no signal; 0.05+ is useful)")
    print(f"  cross-sectional IC  {metrics['mean_cross_sectional_ic']:+.3f}  (ranks items against each other)")

    # Backtest: instant-buy at ask now, instant-sell at bid after the horizon, pay tax.
    # Only trade when the predicted move beats the cost of crossing the spread + tax.
    test = test.assign(pred=pred)
    signal = test["pred"] > test["spread_pct"] + args.tax + args.edge
    trades = non_overlapping(test[signal], gap)
    print("\nBacktest (instant-buy -> instant-sell after horizon, after tax)")
    print(describe("buy every item (baseline)", non_overlapping(test, gap)))
    print(describe("model signal", trades))
    if not trades.empty:
        top = trades.groupby("item")["trade_ret"].agg(["count", "mean"]).sort_values("count", ascending=False)
        print("  most-traded items:\n" + top.head(10).to_string(float_format=lambda v: f"{v:+.2%}"))
    metrics["backtest_trades"] = int(len(trades))
    metrics["backtest_mean_return"] = float(trades["trade_ret"].mean()) if len(trades) else None

    importance = pd.Series(model.booster_.feature_importance("gain"), index=feats).sort_values(ascending=False)
    print("\nFeature importance (gain):\n" + (importance / importance.sum()).round(3).to_string())

    # Refit on everything with the tuned tree count, for use by signals.py
    final = lgb.LGBMRegressor(n_estimators=max(1, model.best_iteration_), **PARAMS)
    final.fit(df[feats], df["y"].clip(lo, hi))

    out = Path(args.out)
    out.mkdir(exist_ok=True)
    final.booster_.save_model(out / "model.txt")
    meta = {"freq": args.freq, "horizon": args.horizon, "tax": args.tax,
            "min_coin_vol": args.min_coin_vol, "features": feats, "metrics": metrics}
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\nSaved model to {out / 'model.txt'}")


if __name__ == "__main__":
    main()
