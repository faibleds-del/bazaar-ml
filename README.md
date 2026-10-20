# Bazaar ML

Predicts short-term Hypixel SkyBlock Bazaar price moves. Costs $0 to run:

- **Collect:** a GitHub Action saves a Bazaar snapshot every ~5 minutes and commits it to this repo.
- **Train:** a Colab or Kaggle notebook clones the repo and trains LightGBM.
- **Use:** `signals.py` ranks items by predicted move after spread and tax. You trade by hand. No macros, which get you banned.

## Setup

1. Create a **public** repo on GitHub (e.g. `bazaar-ml`). Public repos get unlimited Actions minutes; private ones burn through the free quota fast.
2. Push this folder to it (or drag the files into the GitHub web uploader, keeping the `.github/workflows/` folder).
3. On GitHub, open **Actions**, select **collect**, and click **Run workflow** once. After about 30 seconds a `snapshot ...` commit should show up with a file under `data/snapshots/`.
4. Leave it running. The cron then fires on its own.

## Training

Open `notebooks/train.ipynb` in Colab:
`https://colab.research.google.com/github/YOUR_USERNAME/bazaar-ml/blob/main/notebooks/train.ipynb`

Set `REPO` in the first cell and run all cells. On Kaggle, use File → Import Notebook instead, and turn on Internet in settings.

How much data you need:
| Data      | What you can expect |
|-----------|---------------------|
| 1 day     | Smoke test only. The pipeline runs, but the numbers mean nothing. |
| 1–2 weeks | 1h-horizon results start to mean something. |
| 4+ weeks  | Enough to trust 6h horizons and the day-of-week and mayor features. |

## Reading the output

`train.py` prints:
- **RMSE vs the no-change baseline.** If the model doesn't beat "price stays the same," it hasn't learned anything.
- **Rank IC.** This is the correlation between predicted and actual moves. A value of 0.02–0.05 is normal for markets and can still be useful. Above 0.1 is suspicious, so look for leakage.
- **Backtest.** Each trade instant-buys at the ask, instant-sells at the bid after the horizon, and pays the 1.25% tax. This is the pessimistic case. If the model makes money here, it's real. If it doesn't, the signal can still help when you place buy and sell orders, but the backtest can't show that because it can't know your fill times.

## SkyBlock events (`skyblock_calendar.py`)

Hypixel has no events API, and none is needed. The SkyBlock calendar runs on a fixed clock: it started on Jun 11, 2019 at 17:55 UTC, a SkyBlock day lasts 20 minutes, and a year lasts 5 days 4 hours. That means every event can be computed for any timestamp, including old snapshots.

| Event | When | Needs perk |
|---|---|---|
| Spooky Festival | Autumn 29–31 | |
| Season of Jerry | Late Winter 24–26 | |
| New Year Celebration | Late Winter 29–31 | |
| Jerry's Workshop | Late Winter | |
| Traveling Zoo | Early Summer 1–3, Early Winter 1–3 | |
| Hoppity's Hunt | Early Spring 1 – Late Spring 31 | |
| Harvest Feast | Early Autumn 1 – Late Autumn 31 | |
| New mayor | Late Spring 27 | |
| Fishing Festival | 1st–3rd of every month | Fishing Festival (Marina) |
| Mining Fiesta | 1st–7th of Early Summer, Late Summer, Autumn, Early Winter | Mining Fiesta (Cole) |

Features for each event: whether it's running, hours until it starts, hours since it last ended, and how far through it is. The collector also logs the mayor's and minister's perks to `data/mayor.csv`. Those perks decide the gated events above and also act as flags (Diana's Mythological Ritual, Foxy's Extra Event, Diaz's Stock Exchange).

**Seasonal lift** is the most important event feature. For each item, it measures how that item has usually moved over the next 3 hours at this point of the SkyBlock year, using only data from before each moment. Events happen roughly every 5 days, so a month of data contains about 6 of each, enough for the model to learn by itself which items react to which events, like candy around Spooky Festival or gifts around Jerry. Both `train.py` and the evolved bots use it.

```
python skyblock_calendar.py     # today's SkyBlock date and countdowns to every event
```

## Genetic algorithm bots (`evolve.py`)

This is a second approach. It doesn't predict prices. Instead, it evolves complete flipping strategies:

1. It starts with 100 random bots. Each bot's genes cover its entry filters (spread, volume, dip and momentum, order-book imbalance, seasonal/event lift), how it ranks items, order size, how long it leaves orders up, the minimum margin it accepts, its stop-loss, and whether it re-lists when outbid.
2. Each generation, every bot trades the same 3 random 7-day windows of past data. Its fitness is its average return.
3. About half die, with worse bots more likely to die. The survivors breed through crossover and mutation, and a few random newcomers keep the population diverse.
4. The trial windows change every 5 generations, which stops bots from memorizing one stretch of market (the video's Halloween candy problem).
5. At the end, the top 10 bots are re-tested on every training window to pick a champion. The champion is then scored on the newest 25% of data, which no bot ever saw. That score is compared against a hand-written "classic flipper" and against simply holding coins.

```
python evolve.py                                         # evolve and test
python evolve.py --live models/best_bot.json --coins 10e6  # what would it do right now
```

**Simulator limits** (read these before trusting any number):
- Fills are estimated from weekly volume, because the API has no trade-by-trade data. While a bot's order is the best price, it gets `--fill-share` (default 5%) of the instant-buy or instant-sell flow, since lots of other flippers are competing for it. Try `--fill-share 0.02` to stress-test a strategy.
- Bots can't move the market. Big orders on thin items look better than they really are.
- Data is in 10-minute steps, so anything faster, like order wars, is invisible.
- You need about 4 weeks of data before 7-day windows mean much. With less data, the script shrinks the windows automatically.

## Files

| File | Purpose |
|------|---------|
| `collector/collect.py` | Fetches the API and saves the top 500 items by coin volume, plus the current mayor. Standard library only. |
| `.github/workflows/collect.yml` | Runs the collector on a schedule and commits the data. |
| `features.py` | Builds a regular time grid, features (including events and seasonal lift), and targets. |
| `skyblock_calendar.py` | SkyBlock date math and the event schedule. |
| `train.py` | Time-split training, evaluation, backtest, and the saved model. |
| `signals.py` | Scores the latest snapshot with the saved model. |
| `evolve.py` | Genetic algorithm: market simulator, evolution, held-out test, and live picks. |

## Things to know

- **Gaps.** GitHub delays scheduled runs when it's busy, so real spacing is about 5–15 min and sometimes longer. `features.py` bins data into 10-minute slots and forward-fills gaps of up to 30 minutes. Longer gaps stay empty.
- **Repo size.** Each snapshot is about 21 KB, which works out to roughly 100–180 MB a month. The workflow uses a sparse checkout, so runs stay fast as data piles up. After about 6 months, consider moving old months into a GitHub Release.
- **Auto-disable.** GitHub can disable scheduled workflows in repos with no activity for 60 days. If you get that email, click the link to keep it enabled.
- **Tax.** The default is 1.25%. If you have Bazaar Flipper upgrades, pass `--tax 0.01125` or `--tax 0.01`.
