#!/usr/bin/env python3
"""
Download daily OHLCV data from Yahoo Finance (yfinance) and write the long-table
format used by this project:

  date,Open,High,Low,Close,Volume,tic

Users must specify the ticker list with --tickers. The output CSV preserves the
given ticker order: for each trading day, rows are written in the order supplied
to --tickers. The default date range requests available sessions from
2013-01-01 up to, but not including, 2024-01-01, covering 2013-2023.
Yahoo Finance treats the end date as exclusive.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parents[1]
YAHOO_TICKER_ALIASES = {
    # Yahoo Finance uses BRK-B, while papers often write Berkshire Hathaway B
    # shares as BRK.B. Keep BRK.B in the exported CSV for readability.
    "BRK.B": "BRK-B",
}


def _normalize_ticker(s: str) -> str:
    return str(s).strip().upper()


def _yahoo_ticker(tic: str) -> str:
    return YAHOO_TICKER_ALIASES.get(tic, tic)


def _make_curl_session():
    """Use curl_cffi browser impersonation to reduce Yahoo rate-limit failures."""
    try:
        import curl_cffi.requests as curl_requests  # type: ignore[import-untyped]

        for imp in ("chrome", "chrome124", "safari17_0"):
            try:
                return curl_requests.Session(impersonate=imp)
            except Exception:  # noqa: BLE001
                continue
    except ImportError:
        pass
    return None


def _fetch_one(
    tic: str,
    start: str,
    end: str,
    *,
    auto_adjust: bool,
    session,
    retries: int = 4,
) -> pd.DataFrame:
    """Try Ticker.history first, then retry with backoff and fall back to download."""
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            t = yf.Ticker(tic, session=session) if session is not None else yf.Ticker(tic)
            raw = t.history(start=start, end=end, auto_adjust=auto_adjust, repair=False)
            if raw is not None and not raw.empty:
                return raw
        except Exception as e:  # noqa: BLE001
            last_err = e
        time.sleep(1.5 * (attempt + 1))
    # Fall back to yf.download.
    try:
        raw = yf.download(
            tic,
            start=start,
            end=end,
            progress=False,
            auto_adjust=auto_adjust,
            threads=False,
            session=session,
        )
        if raw is not None and not raw.empty:
            return raw
    except Exception as e:  # noqa: BLE001
        last_err = e
    hint = f" last_error={last_err!r}" if last_err else ""
    raise SystemExit(f"No data for {tic}. Check network, proxy, or Yahoo rate limits.{hint}")


def download_panel(
    tickers: list[str],
    start: str,
    end: str,
    sleep_sec: float,
    auto_adjust: bool,
    drop_incomplete_dates: bool,
    session,
) -> pd.DataFrame:
    """The yfinance end date is exclusive; the last row depends on the market calendar."""
    frames: list[pd.DataFrame] = []
    for i, tic in enumerate(tickers):
        if i and sleep_sec > 0:
            time.sleep(sleep_sec)
        yahoo_tic = _yahoo_ticker(tic)
        raw = _fetch_one(yahoo_tic, start, end, auto_adjust=auto_adjust, session=session)
        if raw.empty:
            raise SystemExit(f"No data for {tic}. Check network, ticker symbol, or date range.")
        df = raw.rename(
            columns={
                "Open": "Open",
                "High": "High",
                "Low": "Low",
                "Close": "Close",
                "Volume": "Volume",
            }
        )
        if "Adj Close" in df.columns:
            df = df.drop(columns=["Adj Close"])
        df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
        df.index = pd.to_datetime(df.index).normalize()
        df = df.rename_axis("date").reset_index()
        df["date"] = df["date"].dt.strftime("%Y-%m-%d")
        df["tic"] = tic
        df["_ticker_order"] = i
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    out = out.sort_values(["date", "_ticker_order"]).reset_index(drop=True)
    if drop_incomplete_dates:
        counts = out.groupby("date")["tic"].nunique()
        complete_dates = counts[counts == len(tickers)].index
        out = out[out["date"].isin(complete_dates)].copy()
        out = out.sort_values(["date", "_ticker_order"]).reset_index(drop=True)
    out = out.drop(columns=["_ticker_order"])
    return out


def validate_panel(panel: pd.DataFrame) -> None:
    required_cols = ["date", "Open", "High", "Low", "Close", "Volume", "tic"]
    missing_cols = [c for c in required_cols if c not in panel.columns]
    if missing_cols:
        raise SystemExit(f"Output data is missing columns: {missing_cols}")

    duplicate_date_tic = int(panel.duplicated(["date", "tic"]).sum())
    missing_values = int(panel[required_cols].isna().sum().sum())
    counts = panel.groupby("tic")["date"].nunique().sort_index()

    print("Validation summary:")
    print(f"  rows={len(panel)}")
    print(f"  tickers={len(counts)}")
    print(f"  first_date={panel['date'].min()}")
    print(f"  last_date={panel['date'].max()}")
    print(f"  min_rows_per_ticker={int(counts.min())}")
    print(f"  max_rows_per_ticker={int(counts.max())}")
    print(f"  duplicate_date_tic={duplicate_date_tic}")
    print(f"  missing_values={missing_values}")
    print(f"  rectangular={bool(counts.nunique() == 1 and duplicate_date_tic == 0 and missing_values == 0)}")

    if duplicate_date_tic:
        raise SystemExit("Output data has duplicate (date, tic) rows. Check the download result.")
    if missing_values:
        raise SystemExit("Output data has missing values. Check the download result.")


def main() -> None:
    p = argparse.ArgumentParser(description="Yahoo Finance multi-ticker panel -> FinRL-style long CSV")
    p.add_argument(
        "--tickers",
        type=str,
        required=True,
        help="Comma-separated tickers, for example AAPL,MSFT,NVDA,AMZN,GOOGL,META",
    )
    p.add_argument("--start", type=str, default="2013-01-01")
    p.add_argument(
        "--end",
        type=str,
        default="2024-01-01",
        help="yfinance treats end as exclusive; the default requests sessions before 2024-01-01, covering 2013-2023",
    )
    p.add_argument(
        "--out",
        type=str,
        default="",
        help="Output CSV path. Defaults to data/processed/stock_data_<tickers>_<start>_<end>.csv",
    )
    p.add_argument(
        "--sleep-between-tickers",
        type=float,
        default=2.0,
        help="Sleep seconds between tickers to reduce Yahoo 429 rate limits",
    )
    p.add_argument(
        "--no-curl-impersonate",
        action="store_true",
        help="Disable curl_cffi browser impersonation. Enabled by default when available.",
    )
    p.add_argument(
        "--auto-adjust",
        action="store_true",
        help="Use Yahoo adjusted OHLC. Default keeps raw OHLC; enable this for split-heavy long-horizon panels.",
    )
    p.add_argument(
        "--drop-incomplete-dates",
        action="store_true",
        help="Keep only dates where all tickers are available, producing a rectangular panel.",
    )
    args = p.parse_args()
    tickers = [_normalize_ticker(x) for x in args.tickers.split(",") if x.strip()]
    if not tickers:
        raise SystemExit("tickers is empty")
    if len(set(tickers)) != len(tickers):
        raise SystemExit(f"tickers contains duplicates: {tickers}")

    session = None if args.no_curl_impersonate else _make_curl_session()
    if session is not None:
        print("Using curl_cffi session for Yahoo downloads")
    print("ticker_order=" + ",".join(tickers))
    price_mode = "adjusted_ohlc" if args.auto_adjust else "raw_ohlc"
    print(f"price_mode={price_mode}")
    panel = download_panel(
        tickers,
        args.start,
        args.end,
        args.sleep_between_tickers,
        args.auto_adjust,
        args.drop_incomplete_dates,
        session,
    )
    if args.out.strip():
        out_path = Path(args.out).expanduser()
    else:
        tag = "_".join(tickers)
        s = args.start.replace("-", "")
        e = (pd.Timestamp(args.end) - pd.Timedelta(days=1)).strftime("%Y%m%d")
        out_path = ROOT / "data" / "processed" / f"stock_data_{tag}_{s}_{e}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    validate_panel(panel)
    panel.to_csv(out_path, index=False)
    print(f"Wrote {len(panel)} rows -> {out_path}")


if __name__ == "__main__":
    main()
