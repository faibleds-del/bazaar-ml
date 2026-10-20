"""SkyBlock calendar: dates and recurring events, computed from the clock.

Hypixel has no events API, but none is needed: the SkyBlock calendar runs on a
fixed schedule from a known epoch, so every event's timing can be computed for
any timestamp, including old snapshots.

  1 SkyBlock day   = 20 real minutes
  1 SkyBlock month = 31 days   = 10h 20m real time
  1 SkyBlock year  = 12 months = 5 days 4 hours real time

Mayor-dependent events (Fishing Festival, Mining Fiesta) only switch on when the
perk is active; the collector logs the current perks to data/mayor.csv.

Sources: hypixelskyblock.minecraft.wiki (Time Systems, Calendar, Events pages).

    python skyblock_calendar.py     # print today's SkyBlock date and upcoming events
"""
import time

import numpy as np

EPOCH_MS = 1_560_275_700_000  # 1st of Early Spring, Year 1 = 2019-06-11 17:55 UTC
DAY_MS = 20 * 60 * 1000
MONTH_DAYS = 31
YEAR_DAYS = 12 * MONTH_DAYS
HOURS_PER_DAY = DAY_MS / 3_600_000
MONTHS = ["Early Spring", "Spring", "Late Spring", "Early Summer", "Summer", "Late Summer",
          "Early Autumn", "Autumn", "Late Autumn", "Early Winter", "Winter", "Late Winter"]

# name: ([(start month, start day, end month, end day), ...], perk required or None)
# Months are 1-12 in the order above; day ranges are inclusive.
EVENTS = {
    "spooky_festival": ([(8, 29, 8, 31)], None),
    "season_of_jerry": ([(12, 24, 12, 26)], None),
    "new_year": ([(12, 29, 12, 31)], None),
    "jerrys_workshop": ([(12, 1, 12, 31)], None),
    "traveling_zoo": ([(4, 1, 4, 3), (10, 1, 10, 3)], None),
    "hoppitys_hunt": ([(1, 1, 3, 31)], None),
    "harvest_feast": ([(7, 1, 9, 31)], None),
    "new_mayor": ([(3, 27, 3, 27)], None),  # election closes, new mayor takes office
    "fishing_festival": ([(m, 1, m, 3) for m in range(1, 13)], "Fishing Festival"),
    # Cole's Spring fiesta is bugged and never happens, so only these four
    "mining_fiesta": ([(m, 1, m, 7) for m in (4, 6, 8, 10)], "Mining Fiesta"),
}

# Perks with big market effects but no fixed schedule: exposed as on/off flags
PERK_FLAGS = {
    "perk_mythological_ritual": "Mythological Ritual",  # Diana
    "perk_extra_event": "Extra Event",                  # Foxy
    "perk_stock_exchange": "Stock Exchange",            # Diaz
}

EVENT_FIELDS = ("active", "hours_until", "hours_since", "progress")
CALENDAR_FEATURES = (["sb_day_of_year"]
                     + [f"{e}_{f}" for e in EVENTS for f in EVENT_FIELDS]
                     + list(PERK_FLAGS))


def day_of_year(ts_ms):
    """Fractional SkyBlock day of the year, 0 <= x < 372."""
    return ((np.asarray(ts_ms, dtype=np.float64) - EPOCH_MS) / DAY_MS) % YEAR_DAYS


def sb_date(ts_ms):
    days = int((ts_ms - EPOCH_MS) // DAY_MS)
    year, doy = divmod(days, YEAR_DAYS)
    month, day = divmod(doy, MONTH_DAYS)
    return year + 1, MONTHS[month], day + 1


def _has_perk(perks, perk, n):
    if perks is None:
        return np.zeros(n, dtype=bool)
    return np.array([isinstance(p, str) and perk in p for p in perks], dtype=bool)


def event_features(ts_ms, perks=None):
    """Calendar features for each timestamp (ms).

    For every event: <event>_active (0/1), _hours_until (next start),
    _hours_since (last end) and _progress (0-1 while running, else NaN).
    `perks` is an array of '|'-joined perk names aligned with ts_ms; perk-gated
    events are off (and their timings NaN) when the perk isn't active.
    """
    doy = day_of_year(ts_ms)
    n = doy.shape[0]
    out = {"sb_day_of_year": doy}
    for name, (spans, perk) in EVENTS.items():
        active = np.zeros(n, dtype=bool)
        progress = np.full(n, np.nan)
        until = np.full(n, np.inf)
        since = np.full(n, np.inf)
        for m1, d1, m2, d2 in spans:
            start = (m1 - 1) * MONTH_DAYS + d1 - 1
            end = (m2 - 1) * MONTH_DAYS + d2
            for shift in (-YEAR_DAYS, 0, YEAR_DAYS):  # previous / this / next year
                s, e = start + shift, end + shift
                now = (doy >= s) & (doy < e)
                active |= now
                progress = np.where(now, (doy - s) / (e - s), progress)
                until = np.where(s > doy, np.minimum(until, s - doy), until)
                since = np.where(e <= doy, np.minimum(since, doy - e), since)
        cols = {"active": active.astype(float), "hours_until": until * HOURS_PER_DAY,
                "hours_since": since * HOURS_PER_DAY, "progress": progress}
        if perk is not None:
            on = _has_perk(perks, perk, n)
            cols = {k: np.where(on, v, 0.0 if k == "active" else np.nan) for k, v in cols.items()}
        for field, values in cols.items():
            out[f"{name}_{field}"] = values
    for col, perk in PERK_FLAGS.items():
        out[col] = _has_perk(perks, perk, n).astype(float)
    return out


def summary(ts_ms, perks=""):
    """One line: SkyBlock date, running events, and the next event to start."""
    f = event_features(np.array([ts_ms]), np.array([perks]))
    year, month, day = sb_date(ts_ms)
    title = lambda n: n.replace("_", " ").title()
    active = [title(n) for n in EVENTS if f[f"{n}_active"][0]]
    upcoming = [(f[f"{n}_hours_until"][0], n) for n in EVENTS
                if not f[f"{n}_active"][0] and np.isfinite(f[f"{n}_hours_until"][0])]
    hours, nxt = min(upcoming)
    return (f"SkyBlock {month} {day}, Year {year} | running: {', '.join(active) or 'nothing'} | "
            f"next: {title(nxt)} in {hours:.1f}h")


def main():
    now = time.time() * 1000
    year, month, day = sb_date(now)
    print(f"SkyBlock date: {month} {day}, Year {year}\n")
    f = event_features(np.array([now]), perks=np.array(["|".join(p for _, p in EVENTS.values() if p)]))
    for name, (_, perk) in EVENTS.items():
        label = name.replace("_", " ").title()
        gated = f"  (only with the {perk} perk)" if perk else ""
        if f[f"{name}_active"][0]:
            print(f"  {label:<18} ACTIVE, {f[f'{name}_progress'][0]:.0%} through{gated}")
        else:
            h = f[f"{name}_hours_until"][0]
            print(f"  {label:<18} starts in {int(h)}h {int(h % 1 * 60):02d}m{gated}")


if __name__ == "__main__":
    main()
