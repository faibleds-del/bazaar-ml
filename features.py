"""Load collected snapshots and turn them into model features + targets."""
from pathlib import Path

import numpy as np
import pandas as pd

TAX = 0.0125  # Bazaar sell tax. 1.125% / 1% if you have the Bazaar Flipper upgrades.


def load_snapshots(data_dir="data", days=None):
    """Read every snapshot file (or only the last `days` days) into one DataFrame."""
    root = Path(data_dir) / "snapshots"
    files = sorted(root.glob("*/*/*/*.csv.gz"))
    if days is not None:
        cutoff = (pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=days)).strftime("%Y/%m/%d")
        files = [f for f in files if "/".join(f.parts[-4:-1]) >= cutoff]
    if not files:
        raise SystemExit(f"No snapshots found under {root}. Has the collector run yet?")
    print(f"reading {len(files)} snapshot files...")
    return pd.concat((pd.read_csv(f) for f in files), ignore_index=True)


def load_mayor(data_dir="data"):
    path = Path(data_dir) / "mayor.csv"
    if not path.exists():
        return None
    m = pd.read_csv(path)
    m["time"] = pd.to_datetime(m["ts"], unit="ms", utc=True)
    return m


def to_grid(df, freq="10min", max_gap=3):
    """Put every item on a regular time grid.

    GitHub Actions runs at irregular intervals, so snapshots are bucketed into
    `freq` bins and short gaps (up to `max_gap` bins) are forward-filled.
    Longer gaps stay NaN so lag features don't silently span missing data.
    """
    df = df.assign(time=pd.to_datetime(df["ts"], unit="ms", utc=True).dt.floor(freq))
    df = df.sort_values("ts").groupby(["item", "time"]).last().drop(columns="ts")
    times = df.index.get_level_values("time")
    full = pd.MultiIndex.from_product(
        [df.index.get_level_values("item").unique(),
         pd.date_range(times.min(), times.max(), freq=freq)],
        names=["item", "time"],
    )
    grid = df.reindex(full).groupby(level="item").ffill(limit=max_gap)
    return grid.reset_index()


def _imbalance(a, b):
    total = a + b
    return (a - b) / total.where(total > 0)


def add_features(grid, mayor=None, freq="10min", horizon_hours=1.0, tax=TAX):
    """Add features and targets. Every feature only looks at the past.

    Targets:
      y          log change in mid price over the next `horizon_hours`
      trade_ret  realized return of instant-buying now and instant-selling
                 at the horizon, after tax (what you'd actually make)
    """
    g = grid.sort_values(["item", "time"]).reset_index(drop=True)
    bph = int(pd.Timedelta("1h") / pd.Timedelta(freq))  # bins per hour
    horizon = max(1, round(horizon_hours * bph))

    def by_item(col):
        return g.groupby("item", sort=False)[col]

    def rolling(col, window, how):
        return by_item(col).transform(
            lambda s: getattr(s.rolling(window, min_periods=window // 2), how)())

    # Snapshot features
    g["mid"] = (g["ask"] + g["bid"]) / 2
    g["log_mid"] = np.log(g["mid"])
    g["spread_pct"] = (g["ask"] - g["bid"]) / g["mid"]
    g["book_imbalance"] = _imbalance(g["bid_vol"], g["ask_vol"])
    g["order_imbalance"] = _imbalance(g["bid_orders"], g["ask_orders"])
    g["depth5_imbalance"] = _imbalance(g["bid_depth5"], g["ask_depth5"])
    g["flow_imbalance"] = _imbalance(g["buy_week"], g["sell_week"])
    g["log_coin_vol"] = np.log1p(g["buy_week"] * g["ask"] + g["sell_week"] * g["bid"])
    g["ask_thinness"] = g["ask_qs"] / g["ask"] - 1  # how fast price rises as you buy deeper
    g["bid_thinness"] = 1 - g["bid_qs"] / g["bid"]

    # History features
    for h in (1, 3, 6, 24):
        g[f"ret_{h}h"] = by_item("log_mid").diff(h * bph)
    g["_step_ret"] = by_item("log_mid").diff()
    g["vol_6h"] = rolling("_step_ret", 6 * bph, "std")
    g["vol_24h"] = rolling("_step_ret", 24 * bph, "std")
    g["z_24h"] = (g["log_mid"] - rolling("log_mid", 24 * bph, "mean")) / rolling("log_mid", 24 * bph, "std")
    g["spread_rel_24h"] = g["spread_pct"] / rolling("spread_pct", 24 * bph, "mean")
    g["coin_vol_chg_6h"] = by_item("log_coin_vol").diff(6 * bph)
    g["book_imb_chg_1h"] = by_item("book_imbalance").diff(bph)
    g["hour"] = g["time"].dt.hour
    g["dow"] = g["time"].dt.dayofweek
    g = g.drop(columns="_step_ret")

    # Targets (future values, never used as features)
    g["y"] = by_item("log_mid").shift(-horizon) - g["log_mid"]
    g["trade_ret"] = by_item("bid").shift(-horizon) * (1 - tax) / g["ask"] - 1

    if mayor is not None and len(mayor):
        g = pd.merge_asof(g.sort_values("time"), mayor[["time", "mayor"]].sort_values("time"),
                          on="time", direction="backward")
        g = g.sort_values(["item", "time"]).reset_index(drop=True)
        g["mayor"] = g["mayor"].astype("category")

    return g.replace([np.inf, -np.inf], np.nan)


FEATURES = [
    "spread_pct", "book_imbalance", "order_imbalance", "depth5_imbalance",
    "flow_imbalance", "log_coin_vol", "ask_thinness", "bid_thinness",
    "ret_1h", "ret_3h", "ret_6h", "ret_24h", "vol_6h", "vol_24h", "z_24h",
    "spread_rel_24h", "coin_vol_chg_6h", "book_imb_chg_1h", "hour", "dow",
]


def feature_columns(df):
    return FEATURES + (["mayor"] if "mayor" in df.columns else [])
