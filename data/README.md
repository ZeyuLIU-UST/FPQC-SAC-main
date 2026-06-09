# Data Layout

This repository does not commit full market datasets by default. The fixed
Yahoo Finance panels are rebuilt under the following structure:

```text
data/
  raw/        # downloaded vendor files, unchanged
  processed/  # normalized FinRL-style long tables
```

## Expected Processed Schema

Training configs expect a CSV with one row per `(date, tic)` pair:

```text
date,Open,High,Low,Close,Volume,tic
```

The fixed data panels are:

```text
data/processed/repro_mainstream_tech_market_index_portfolio_2013_2023.csv
data/processed/repro_defensive_blue_chip_portfolio_2013_2023.csv
data/processed/repro_high_volatility_growth_portfolio_2013_2023.csv
```

The fixed seed list used by the reproduction scripts is
`configs/seeds_repro_20.txt`.

## Rebuilding The Fixed Panels

Use the one-command downloader:

```bash
bash scripts/download_reproducibility_data.sh
```

The fixed groups are:

```text
mainstream_tech_market_index: mainstream tech and market-index portfolio; AAPL, AMZN, GOOGL, MSFT, QQQ, SPY
defensive_blue_chip: Defensive Blue-chip portfolio with traditional economy and value stocks; BAC, JNJ, JPM, PG, WMT, XOM
high_volatility_growth: High-Volatility Growth portfolio with aggressive thematic assets; AVGO, BRK.B, META, NFLX, NVDA, TSLA
```

The ticker order is part of the reproduction protocol. The downloader writes
rows by ascending date, and within each date it preserves the order shown above.
For `BRK.B`, the downloader queries Yahoo Finance as `BRK-B` and writes `BRK.B`
back to the output CSV.

Yahoo's `end` date is exclusive. The default
`--start 2013-01-01 --end 2024-01-01` requests all available sessions from 2013
through the end of 2023. The last row depends on the market calendar and may be
`2023-12-29` for US stocks.

The downloader prints a validation summary after downloading:

```text
rows, tickers, first_date, last_date, min_rows_per_ticker,
max_rows_per_ticker, duplicate_date_tic, missing_values, rectangular
```

For FinRL panel experiments, check that:

- Required columns are exactly available as `date,Open,High,Low,Close,Volume,tic`.
- `duplicate_date_tic = 0`.
- `missing_values = 0`.
- `rectangular = True` when every ticker should appear on every trading day.

The fixed downloader uses adjusted OHLC and rectangular panels:

```bash
python scripts/download_yahoo_stock_panel.py \
  --tickers AAPL,AMZN,GOOGL,MSFT,QQQ,SPY \
  --start 2013-01-01 \
  --end 2024-01-01 \
  --auto-adjust \
  --drop-incomplete-dates \
  --out data/processed/repro_mainstream_tech_market_index_portfolio_2013_2023.csv
```

Keep the price mode consistent across all algorithms in the same comparison.

## Rebuilding the Local VIX File

The training stack expects a CSV at `data/raw/vix_panel.csv`. Build it with the
same Yahoo downloader:

```bash
python scripts/download_yahoo_stock_panel.py \
  --tickers VIXY \
  --start 2013-01-01 \
  --end 2024-01-01 \
  --out data/raw/vix_panel.csv
```

The training code only requires `date` and `Close`; the downloader also writes
`Open,High,Low,Volume,tic`, which is accepted. Use `VIXY` if you want the
tradable volatility ETF proxy. If your experiment uses the CBOE VIX index
instead, replace `VIXY` with `^VIX` and quote it in shells that treat `^` as a
special character.

Keep the VIX date range at least as wide as the train/test period in your
configs. For the default split, `--start 2013-01-01 --end 2024-01-01` covers the
2013-2023 range. If your config evaluates later dates, set `--end` to the day
after the last date you need because Yahoo treats `end` as exclusive.
