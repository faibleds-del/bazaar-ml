"""Load collected snapshots and turn them into model features + targets."""
from pathlib import Path

import numpy as np
import pandas as pd

from skyblock_calendar import CALENDAR_FEATURES, YEAR_DAYS, day_of_year, event_features

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
    m = pd.read_csv(path, keep_default_na=False)
    m["time"] = pd.to_datetime(m["ts"], unit="ms", utc=True)
    if "perks" not in m.columns:
        m["perks"] = ""
    return m


def _to_ms(times):
    return np.asarray((times - pd.Timestamp(0, tz="UTC")) // pd.Timedelta("1ms"), dtype=np.int64)


def seasonal_lift(g, bph, hold_hours=3, bucket_days=6, prior=50):
    """How this item has usually moved over the next `hold_hours` at this point
    of the SkyBlock year, learned only from data before each moment (no look-ahead).

    Every calendar event sits at a fixed point of the SkyBlock year, which
    repeats every 5d 4h real time, so this picks up item-specific event effects
    (candy around Spooky Festival, gifts around Jerry, ...) without hard-coding
    which items each event moves. `prior` shrinks thin estimates toward zero.
    """
    lm = g.pivot(index="time", columns="item", values="log_mid")
    h = int(hold_hours * bph)
    fwd = (lm.shift(-h) - lm).to_numpy()  # the move starting at row s is known at row s + h
    buckets = (day_of_year(_to_ms(lm.index)) // bucket_days).astype(int)
    n_buckets = int(np.ceil(YEAR_DAYS / bucket_days))
    sums = np.zeros((n_buckets, lm.shape[1]))
    counts = np.zeros((n_buckets, lm.shape[1]))
    out = np.zeros(lm.shape)
    for t in range(len(lm)):
        s = t - h
        if s >= 0:
            ok = np.isfinite(fwd[s])
            sums[buckets[s], ok] += fwd[s, ok]
            counts[buckets[s], ok] += 1
        out[t] = sums[buckets[t]] / (counts[buckets[t]] + prior)
    return (pd.DataFrame(out, index=lm.index, columns=lm.columns).reset_index()
            .melt(id_vars="time", var_name="item", value_name="seasonal_lift"))


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

    # Item-specific seasonality (event effects), learned causally
    g = g.merge(seasonal_lift(g, bph), on=["time", "item"], how="left")

    # Mayor + SkyBlock calendar events, computed once per time step
    tf = pd.DataFrame({"time": pd.Series(g["time"].unique()).sort_values(ignore_index=True)})
    if mayor is not None and len(mayor):
        tf = pd.merge_asof(tf, mayor[["time", "mayor", "minister", "perks"]].sort_values("time"),
                           on="time", direction="backward")
    perks = tf["perks"].to_numpy() if "perks" in tf.columns else None
    tf = pd.concat([tf, pd.DataFrame(event_features(_to_ms(tf["time"]), perks))], axis=1)
    g = g.merge(tf.drop(columns="perks", errors="ignore"), on="time", how="left")
    for col in ("mayor", "minister"):
        if col in g.columns:
            g[col] = g[col].astype("category")

    return g.replace([np.inf, -np.inf], np.nan)


FEATURES = [
    "spread_pct", "book_imbalance", "order_imbalance", "depth5_imbalance",
    "flow_imbalance", "log_coin_vol", "ask_thinness", "bid_thinness",
    "ret_1h", "ret_3h", "ret_6h", "ret_24h", "vol_6h", "vol_24h", "z_24h",
    "spread_rel_24h", "coin_vol_chg_6h", "book_imb_chg_1h", "hour", "dow",
    "seasonal_lift",
] + CALENDAR_FEATURES


def feature_columns(df):
    return FEATURES + [c for c in ("mayor", "minister") if c in df.columns]
