#!/usr/bin/env bash
set -euo pipefail

# Download the fixed reproduction panels. The ticker order in each command is
# part of the experiment definition and is preserved in the CSV rows.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
START="${START:-2013-01-01}"
END="${END:-2024-01-01}"
SLEEP_BETWEEN_TICKERS="${SLEEP_BETWEEN_TICKERS:-2.0}"

cd "$ROOT"

mkdir -p data/processed data/raw

download_panel() {
  local group_id="$1"
  local tickers="$2"
  local out="$3"

  echo "=== download ${group_id}: ${tickers} ==="
  "$PYTHON_BIN" scripts/download_yahoo_stock_panel.py \
    --tickers "$tickers" \
    --start "$START" \
    --end "$END" \
    --sleep-between-tickers "$SLEEP_BETWEEN_TICKERS" \
    --auto-adjust \
    --drop-incomplete-dates \
    --out "$out"
}

download_panel "mainstream_tech_market_index_portfolio" \
  "AAPL,AMZN,GOOGL,MSFT,QQQ,SPY" \
  "data/processed/repro_mainstream_tech_market_index_portfolio_2013_2023.csv"

download_panel "defensive_blue_chip_portfolio" \
  "BAC,JNJ,JPM,PG,WMT,XOM" \
  "data/processed/repro_defensive_blue_chip_portfolio_2013_2023.csv"

download_panel "high_volatility_growth_portfolio" \
  "AVGO,BRK.B,META,NFLX,NVDA,TSLA" \
  "data/processed/repro_high_volatility_growth_portfolio_2013_2023.csv"

echo "=== download vix proxy: VIXY ==="
"$PYTHON_BIN" scripts/download_yahoo_stock_panel.py \
  --tickers VIXY \
  --start "$START" \
  --end "$END" \
  --sleep-between-tickers "$SLEEP_BETWEEN_TICKERS" \
  --auto-adjust \
  --drop-incomplete-dates \
  --out data/raw/vix_panel.csv

echo "=== fixed reproduction data ready ==="
