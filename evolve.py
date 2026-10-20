"""Evolve Bazaar flipping bots with a genetic algorithm.

    python evolve.py                         # evolve, then test the winner on held-out data
    python evolve.py --gens 50 --pop 60      # quicker run
    python evolve.py --resume                # continue after a Colab disconnect
    python evolve.py --live models/best_bot.json --coins 10e6   # what would the bot do right now?

Each bot is a vector of genes (entry filters, ranking weights, order timing,
risk limits). Every generation:
  1. each bot trades the same few randomly chosen windows of past data
  2. fitness = average log return across those windows
  3. roughly half die, worse bots more likely
  4. survivors breed (uniform crossover) and mutate to refill the population
The windows rotate every few generations so bots can't memorize one stretch
of market (the video switched periods every 20 generations; we do it more
often). The newest slice of data is never seen during evolution and is used
only for the final score.

Simulation model (deliberately simple; see README for its limits):
  - buy order placed at best bid + 0.1, sell offer at best ask - 0.1
  - while our order is the best price, it fills from the instant-sell (or
    instant-buy) flow implied by weekly volume, times --fill-share (default 5%)
    to account for all the other flippers competing for the same flow
  - if the market moves through our price, the order fills completely
  - 1.25% tax on every sale, 14 open orders max, 71,680 units max per order
"""
import argparse
import json
import multiprocessing as mp
import os
import time
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from features import TAX, add_features, load_mayor, load_snapshots, to_grid
from skyblock_calendar import summary

TICK = 0.1
ORDER_SLOTS = 14
MAX_ORDER_UNITS = 71_680

# name, low, high, scale. Bots store genes as numbers in [0, 1]; decode() maps them.
GENES = [
    ("min_spread", 0.002, 0.3, "log"),      # only flip items whose spread is at least this
    ("max_spread", 0.01, 1.0, "log"),       # ...and at most this (huge spreads = dead items)
    ("min_coin_vol", 1e7, 1e11, "log"),     # weekly coin volume floor
    ("max_z", -3.0, 3.0, "lin"),            # only buy when price is below its 24h mean by this many sd
    ("min_momentum", -0.2, 0.2, "lin"),     # only buy when 6h price change is above this
    ("min_book_imb", -1.0, 1.0, "lin"),     # only buy when (buy orders - sell offers) imbalance is above this
    ("min_season", -0.05, 0.05, "lin"),     # only buy when the item usually rises at this point of the
                                            # SkyBlock year (events), by at least this much
    ("w_spread", -1.0, 1.0, "lin"),         # ranking weights for choosing among candidates
    ("w_momentum", -1.0, 1.0, "lin"),
    ("w_dip", -1.0, 1.0, "lin"),
    ("w_book", -1.0, 1.0, "lin"),
    ("w_volume", -1.0, 1.0, "lin"),
    ("w_flow", -1.0, 1.0, "lin"),
    ("w_season", -1.0, 1.0, "lin"),
    ("alloc", 0.02, 0.5, "log"),            # fraction of total coins per order
    ("max_positions", 1, ORDER_SLOTS, "int"),
    ("buy_ttl_h", 0.2, 6.0, "log"),         # cancel unfilled buy orders after this many hours
    ("sell_ttl_h", 0.2, 24.0, "log"),       # instant-sell leftovers after this many hours
    ("chase", 0.0, 1.0, "bool"),            # re-list at the new best price when outbid/undercut
    ("min_margin", 0.0, 0.1, "lin"),        # never list a sell offer below cost + this, after tax
    ("stop_loss", 0.02, 0.5, "log"),        # instant-sell if bid drops this far below cost
]
WEIGHTS = ["w_spread", "w_momentum", "w_dip", "w_book", "w_volume", "w_flow", "w_season"]
N_GENES = len(GENES)

# A hand-written "normal flipper" to compare the evolved bots against
CLASSIC_FLIPPER = dict(
    min_spread=0.03, max_spread=0.3, min_coin_vol=1e9, max_z=3.0, min_momentum=-0.2,
    min_book_imb=-1.0, min_season=-0.05, w_spread=1.0, w_momentum=0.0, w_dip=0.0, w_book=0.0,
    w_volume=0.0, w_flow=0.0, w_season=0.0, alloc=0.1, max_positions=10, buy_ttl_h=2.0, sell_ttl_h=12.0, chase=1.0,
    min_margin=0.01, stop_loss=0.15,
)

MARKET = None  # set in main(); worker processes inherit it via fork


def decode(v):
    g = {}
    for x, (name, lo, hi, kind) in zip(v, GENES):
        if kind == "log":
            g[name] = lo * (hi / lo) ** x
        elif kind == "int":
            g[name] = int(round(lo + x * (hi - lo)))
        elif kind == "bool":
            g[name] = bool(x >= 0.5)
        else:
            g[name] = lo + x * (hi - lo)
    g["weights"] = np.array([g[w] for w in WEIGHTS])
    return g


def encode(genes):
    v = np.empty(N_GENES)
    for k, (name, lo, hi, kind) in enumerate(GENES):
        val = float(genes[name])
        v[k] = np.log(val / lo) / np.log(hi / lo) if kind == "log" else (val - lo) / (hi - lo)
    return np.clip(v, 0, 1)


def describe(g):
    ranked = sorted(WEIGHTS, key=lambda w: -abs(g[w]))[:2]
    prefs = ", ".join(f"{'high' if g[w] > 0 else 'low'} {w[2:]}" for w in ranked)
    return (
        f"Buys items with spread {g['min_spread']:.1%}-{g['max_spread']:.1%}, weekly volume >= "
        f"{g['min_coin_vol'] / 1e6:,.0f}M coins, 24h z-score < {g['max_z']:+.1f}, 6h move > "
        f"{g['min_momentum']:+.1%}, book imbalance > {g['min_book_imb']:+.2f}, seasonal lift > "
        f"{g['min_season']:+.1%}. Prefers {prefs}.\n"
        f"  Up to {g['max_positions']} orders at {g['alloc']:.0%} of coins each. Cancels buys after "
        f"{g['buy_ttl_h']:.1f}h, sells at >= {g['min_margin']:.1%} profit, dumps after "
        f"{g['sell_ttl_h']:.1f}h or a {g['stop_loss']:.0%} drop. "
        f"{'Re-lists' if g['chase'] else 'Does not re-list'} when outbid."
    )


# --------------------------------------------------------------------------- market

def _xs_zscore(a):
    """Standardize each column across items at each time step (NaN -> 0)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mu = np.nanmean(a, axis=1, keepdims=True)
        sd = np.nanstd(a, axis=1, keepdims=True)
    return np.nan_to_num((a - mu) / np.where(sd > 0, sd, 1.0))


def build_market(data_dir, freq, days=None, min_presence=0.5):
    df = add_features(to_grid(load_snapshots(data_dir, days=days), freq), None, freq)

    def wide(col):
        return df.pivot(index="time", columns="item", values=col)

    ask = wide("ask")
    items = ask.columns[ask.notna().mean() >= min_presence]

    def arr(col):
        return wide(col).reindex(columns=items).to_numpy(dtype=float)

    bph = int(pd.Timedelta("1h") / pd.Timedelta(freq))
    per_bin = 7 * 24 * bph
    m = SimpleNamespace(
        times=ask.index, items=list(items), bph=bph,
        ask=arr("ask"), bid=arr("bid"), spread=arr("spread_pct"),
        z=arr("z_24h"), mom=arr("ret_6h"), imb=arr("book_imbalance"), season=arr("seasonal_lift"),
        buy_flow=arr("buy_week") / per_bin,    # instant-buys per bin -> fills our sell offers
        sell_flow=arr("sell_week") / per_bin,  # instant-sells per bin -> fills our buy orders
    )
    m.coin_vol = arr("buy_week") * m.ask + arr("sell_week") * m.bid
    m.valid = np.isfinite(m.ask) & np.isfinite(m.bid) & (m.ask > m.bid)
    m.rank_feats = np.stack([
        _xs_zscore(m.spread), _xs_zscore(m.mom), _xs_zscore(-m.z), _xs_zscore(m.imb),
        _xs_zscore(np.log1p(m.coin_vol)), _xs_zscore(arr("flow_imbalance")), _xs_zscore(m.season),
    ], axis=-1).astype(np.float32)
    m.n_times, m.n_items = m.ask.shape
    return m


# --------------------------------------------------------------------------- simulation

def _equity(cash, positions, tax):
    total = cash
    for p in positions:
        b = p["last_bid"]
        if p["side"] == "buy":
            total += (p["qty"] - p["filled"]) * p["price"] + p["filled"] * b * (1 - tax)
        else:
            total += (p["qty"] - p["filled"]) * b * (1 - tax)
    return total


def simulate(g, t0, t1, m):
    """Run one bot over time steps [t0, t1). Returns (final coins, [(item index, profit), ...])."""
    tax, share = m.tax, m.fill_share
    buy_ttl = max(1, round(g["buy_ttl_h"] * m.bph))
    sell_ttl = max(1, round(g["sell_ttl_h"] * m.bph))
    slots = min(g["max_positions"], ORDER_SLOTS)
    floor_mult = (1 + g["min_margin"]) / (1 - tax)
    cash = m.coins
    positions, trades = [], []
    held = np.zeros(m.n_items, dtype=bool)

    for t in range(t0, t1):
        ask, bid, valid = m.ask[t], m.bid[t], m.valid[t]

        # 1. Manage open orders
        still_open = []
        for p in positions:
            i = p["i"]
            p["age"] += 1
            if not valid[i]:
                still_open.append(p)
                continue
            a, b = ask[i], bid[i]
            p["last_bid"] = b

            if p["side"] == "buy":
                left = p["qty"] - p["filled"]
                if a <= p["price"]:
                    fill = left
                elif b <= p["price"]:
                    fill = min(left, share * m.sell_flow[t, i])
                else:  # outbid
                    fill = 0.0
                    if g["chase"] and (a - b) * 2 / (a + b) >= g["min_spread"]:
                        extra = left * (b + TICK - p["price"])
                        if extra <= cash:
                            cash -= extra
                            p["price"] = b + TICK
                p["filled"] += fill
                p["cost"] += fill * p["price"]
                if p["filled"] >= p["qty"] - 1e-9 or p["age"] >= buy_ttl:
                    cash += (p["qty"] - p["filled"]) * p["price"]  # refund the unfilled part
                    if p["filled"] > 1e-9:
                        basis = p["cost"] / p["filled"]
                        p.update(side="sell", qty=p["filled"], filled=0.0, basis=basis,
                                 floor=basis * floor_mult, age=0, proceeds=0.0,
                                 price=max(a - TICK, basis * floor_mult))
                        still_open.append(p)
                    else:
                        held[i] = False
                else:
                    still_open.append(p)

            else:  # sell offer
                left = p["qty"] - p["filled"]
                if b >= p["price"]:
                    fill = left
                elif a >= p["price"]:
                    fill = min(left, share * m.buy_flow[t, i])
                else:  # undercut
                    fill = 0.0
                    if g["chase"]:
                        p["price"] = min(p["price"], max(a - TICK, p["floor"]))
                got = fill * p["price"] * (1 - tax)
                cash += got
                p["proceeds"] += got
                p["filled"] += fill
                left -= fill
                if left > 1e-9 and (b < p["basis"] * (1 - g["stop_loss"]) or p["age"] >= sell_ttl):
                    got = left * b * (1 - tax)
                    cash += got
                    p["proceeds"] += got
                    left = 0.0
                if left <= 1e-9:
                    held[i] = False
                    trades.append((i, p["proceeds"] - p["basis"] * p["qty"]))
                else:
                    still_open.append(p)
        positions = still_open

        # 2. Place new buy orders
        free = slots - len(positions)
        if free <= 0 or cash < 1:
            continue
        spread = m.spread[t]
        ok = (valid & ~held
              & (spread >= g["min_spread"]) & (spread <= g["max_spread"])
              & (m.coin_vol[t] >= g["min_coin_vol"]) & (m.z[t] < g["max_z"])
              & (m.mom[t] > g["min_momentum"]) & (m.imb[t] > g["min_book_imb"])
              & (m.season[t] >= g["min_season"]))
        idx = np.flatnonzero(ok)
        if idx.size == 0:
            continue
        if idx.size > free:
            idx = idx[np.argsort(-(m.rank_feats[t, idx] @ g["weights"]))[:free]]
        equity = _equity(cash, positions, tax)
        for i in idx:
            price = bid[i] + TICK
            qty = min(min(cash, g["alloc"] * equity) // price, MAX_ORDER_UNITS)
            if qty < 1:
                continue
            cash -= qty * price
            positions.append(dict(i=i, side="buy", price=price, qty=qty, filled=0.0,
                                  cost=0.0, age=0, last_bid=bid[i]))
            held[i] = True

    return _equity(cash, positions, tax), trades


def fitness(job):
    v, windows = job
    g = decode(v)
    log_rets, n_trades = [], 0
    for t0, t1 in windows:
        final, trades = simulate(g, t0, t1, MARKET)
        log_rets.append(np.log(max(final, 1.0) / MARKET.coins))
        n_trades += len(trades)
    return float(np.mean(log_rets)), n_trades / len(windows)


# --------------------------------------------------------------------------- evolution

def next_generation(pop, fit, rng, elite=2, mut_rate=0.15, mut_sd=0.1, immigrants=0.05):
    """Kill ~half (worse bots more likely), breed survivors back up to full size."""
    size = len(pop)
    order = np.argsort(-fit)
    rest = order[elite:]
    survive = rest[rng.random(len(rest)) < np.linspace(0.9, 0.1, len(rest))]
    keep = np.concatenate([order[:elite], survive])  # still in rank order
    survivors = pop[keep]
    parent_p = np.linspace(1.0, 0.2, len(keep))
    parent_p /= parent_p.sum()

    n_children = size - len(keep) - int(immigrants * size)
    children = np.empty((max(n_children, 0), N_GENES))
    for c in range(len(children)):
        a, b = rng.choice(len(keep), size=2, replace=False, p=parent_p)
        child = np.where(rng.random(N_GENES) < 0.5, survivors[a], survivors[b])
        child = child + (rng.random(N_GENES) < mut_rate) * rng.normal(0, mut_sd, N_GENES)
        reset = rng.random(N_GENES) < 0.02
        child[reset] = rng.random(reset.sum())
        children[c] = np.clip(child, 0, 1)
    newcomers = rng.random((size - len(keep) - len(children), N_GENES))
    return np.vstack([survivors, children, newcomers])


def evaluate(pop, windows, pool, cache):
    jobs = [(v, windows) for v in pop]
    keys = [(v.tobytes(), windows) for v in pop]
    todo = [k for k in range(len(pop)) if keys[k] not in cache]
    results = (pool.map(fitness, [jobs[k] for k in todo], chunksize=4) if pool
               else [fitness(jobs[k]) for k in todo])
    for k, r in zip(todo, results):
        cache[keys[k]] = r
    fit = np.array([cache[k][0] for k in keys])
    trades = np.array([cache[k][1] for k in keys])
    return fit, trades


def window_report(g, windows, m):
    rets, by_item = [], {}
    for t0, t1 in windows:
        final, trades = simulate(g, t0, t1, m)
        rets.append(final / m.coins - 1)
        for i, profit in trades:
            by_item[m.items[i]] = by_item.get(m.items[i], 0.0) + profit
    return np.array(rets), pd.Series(by_item, dtype=float).sort_values(ascending=False)


def fmt_pct(rets):
    return " ".join(f"{r:+.1%}" for r in rets)


def live(args, m):
    bot = json.loads(Path(args.live).read_text())
    g = decode(encode(bot["genes"]))
    t = m.n_times - 1
    mayor = load_mayor(args.data)
    perks = mayor["perks"].iloc[-1] if mayor is not None else ""
    print(f"Bot: {describe(g)}\nData as of {m.times[t]:%Y-%m-%d %H:%M} UTC")
    print(summary(int(m.times[t].timestamp() * 1000), perks) + "\n")
    spread = m.spread[t]
    ok = (m.valid[t] & (spread >= g["min_spread"]) & (spread <= g["max_spread"])
          & (m.coin_vol[t] >= g["min_coin_vol"]) & (m.z[t] < g["max_z"])
          & (m.mom[t] > g["min_momentum"]) & (m.imb[t] > g["min_book_imb"])
          & (m.season[t] >= g["min_season"]))
    idx = np.flatnonzero(ok)
    if idx.size == 0:
        print("Nothing passes this bot's filters right now. It would sit on its coins.")
        return
    idx = idx[np.argsort(-(m.rank_feats[t, idx] @ g["weights"]))][: g["max_positions"]]
    rows = []
    for i in idx:
        buy = m.bid[t, i] + TICK
        qty = min(g["alloc"] * args.coins // buy, MAX_ORDER_UNITS)
        floor = buy * (1 + g["min_margin"]) / (1 - m.tax)
        rows.append({"item": m.items[i], "buy order @": buy, "qty": int(qty),
                     "sell offer @": max(m.ask[t, i] - TICK, floor), "spread": spread[i],
                     "stop-loss @": buy * (1 - g["stop_loss"])})
    print(pd.DataFrame(rows).to_string(index=False, formatters={
        "buy order @": "{:,.1f}".format, "sell offer @": "{:,.1f}".format,
        "stop-loss @": "{:,.1f}".format, "spread": "{:.1%}".format}))
    print(f"\nCancel unfilled buys after {g['buy_ttl_h']:.1f}h; instant-sell leftovers after "
          f"{g['sell_ttl_h']:.1f}h.")


def main():
    global MARKET
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--freq", default="10min")
    ap.add_argument("--pop", type=int, default=100)
    ap.add_argument("--gens", type=int, default=100)
    ap.add_argument("--window-days", type=float, default=7.0, help="length of each trading trial")
    ap.add_argument("--windows", type=int, default=3, help="trial windows per generation")
    ap.add_argument("--rotate-every", type=int, default=5, help="new trial windows every N generations")
    ap.add_argument("--test-frac", type=float, default=0.25, help="newest data held out for the final test")
    ap.add_argument("--coins", type=float, default=10e6)
    ap.add_argument("--fill-share", type=float, default=0.05,
                    help="share of instant-buy/sell flow our top-of-book order gets (lots of competing flippers)")
    ap.add_argument("--tax", type=float, default=TAX)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--out", default="models")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--live", metavar="BOT_JSON", help="show what a saved bot would do right now")
    args = ap.parse_args()

    m = build_market(args.data, args.freq)  # full history, also in --live: seasonal_lift needs past years
    m.coins, m.fill_share, m.tax = args.coins, args.fill_share, args.tax
    MARKET = m
    if args.live:
        return live(args, m)

    out = Path(args.out)
    out.mkdir(exist_ok=True)
    days = m.n_times / (24 * m.bph)
    train_end = int(m.n_times * (1 - args.test_frac))
    win = int(args.window_days * 24 * m.bph)
    if train_end < 2 * win:
        win = train_end // 2
        print(f"Only {days:.1f} days of data: shrinking trial windows to {win / (24 * m.bph):.1f} days.")
    if win < 24 * m.bph:
        raise SystemExit("Need at least ~3 days of data to evolve anything meaningful.")
    test_windows = tuple((s, s + win) for s in range(train_end, m.n_times - win + 1, win)) \
        or ((train_end, m.n_times),)
    print(f"{days:.1f} days, {m.n_items} items. Evolving on the first {train_end / (24 * m.bph):.1f} days, "
          f"testing on the last {(m.n_times - train_end) / (24 * m.bph):.1f}.")

    rng = np.random.default_rng(args.seed)
    ckpt = out / "evolution_checkpoint.npz"
    start_gen = 0
    pop = rng.random((args.pop, N_GENES))
    if args.resume and ckpt.exists():
        saved = np.load(ckpt)
        pop, start_gen = saved["pop"], int(saved["gen"]) + 1
        rng = np.random.default_rng(args.seed + start_gen)
        print(f"Resuming from generation {start_gen}")
        if start_gen >= args.gens:
            raise SystemExit(f"Checkpoint already reached generation {start_gen - 1}; pass a higher --gens.")

    use_fork = args.workers > 1 and "fork" in mp.get_all_start_methods()
    pool = mp.get_context("fork").Pool(args.workers) if use_fork else None
    cache = {}
    windows = None
    history_csv = out / "evolution_history.csv"
    if not (args.resume and start_gen):
        history_csv.unlink(missing_ok=True)
    try:
        for gen in range(start_gen, args.gens):
            started = time.time()
            if windows is None or gen % args.rotate_every == 0:
                starts = np.sort(rng.integers(0, train_end - win + 1, size=args.windows))
                windows = tuple((int(s), int(s) + win) for s in starts)
                cache.clear()
            fit, trades = evaluate(pop, windows, pool, cache)
            best = int(np.argmax(fit))
            pd.DataFrame([{"gen": gen, "best": np.expm1(fit[best]), "median": np.expm1(np.median(fit)),
                           "profitable_bots": int((fit > 0).sum()), "best_trades": trades[best]}]
                         ).to_csv(history_csv, mode="a", index=False, header=not history_csv.exists())
            print(f"gen {gen:3d} | best {np.expm1(fit[best]):+7.1%} | median {np.expm1(np.median(fit)):+7.1%} | "
                  f"{(fit > 0).sum():3d}/{len(pop)} profitable | best bot {trades[best]:.0f} trades/window | "
                  f"{time.time() - started:.0f}s")
            if gen % 10 == 0 or gen == args.gens - 1:
                print("        " + describe(decode(pop[best])).replace("\n", "\n        "))
            if gen < args.gens - 1:
                pop = next_generation(pop, fit, rng)
            np.savez(ckpt, pop=pop, gen=gen)

        # Pick the final champion by re-testing the top 10 on many training windows,
        # not just the last few, so one lucky window can't crown it.
        robust_windows = tuple((s, s + win) for s in range(0, train_end - win + 1, max(1, win // 2)))
        top = np.argsort(-fit)[:10]
        robust, _ = evaluate(pop[top], robust_windows, pool, {})
        champion = decode(pop[top[int(np.argmax(robust))]])
    finally:
        if pool:
            pool.close()

    classic = decode(encode(CLASSIC_FLIPPER))
    print("\n=== Held-out test (data the bots never saw) ===")
    results = {}
    for name, g in (("evolved champion", champion), ("classic flipper", classic)):
        train_rets, _ = window_report(g, robust_windows, m)
        test_rets, by_item = window_report(g, test_windows, m)
        results[name] = {"train_mean": float(train_rets.mean()), "test": test_rets.tolist()}
        print(f"{name:<17} train avg {train_rets.mean():+6.1%} per {win / (24 * m.bph):.1f}d | "
              f"test {fmt_pct(test_rets)}")
        if name == "evolved champion" and len(by_item):
            print("  best items:  " + ", ".join(f"{k} {v / 1e6:+.1f}M" for k, v in by_item.head(5).items()))
            print("  worst items: " + ", ".join(f"{k} {v / 1e6:+.1f}M" for k, v in by_item.tail(3).items()))
    print(f"holding coins     0.0%\n\nChampion: {describe(champion)}")
    print("\nIf the champion beats the classic flipper on train but not on test, it overfit.")

    genes = {name: (bool(champion[name]) if kind == "bool" else float(champion[name]))
             for name, _, _, kind in GENES}
    (out / "best_bot.json").write_text(json.dumps(
        {"genes": genes, "description": describe(champion), "results": results,
         "settings": {k: vars(args)[k] for k in ("freq", "coins", "fill_share", "tax", "window_days")}},
        indent=2))
    print(f"Saved {out / 'best_bot.json'}")


if __name__ == "__main__":
    main()
