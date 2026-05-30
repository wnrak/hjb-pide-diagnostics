#!/usr/bin/env python
"""Convert a user-supplied S&P 500 daily-close series into the daily-return file
the calibration expects.

This repository deliberately does NOT ship S&P 500 index levels or derived
returns (see data/README.md). Supply your own lawfully obtained daily close
data and run this script to produce ``sp500_returns_local.csv`` locally.

Expected input CSV columns (see ``sp500_schema_example.csv`` for the format;
that example file contains synthetic rows only):

    Date,Close
    2010-01-04,1132.99
    2010-01-05,1136.52
    ...

Usage:
    python prepare_sp500_returns.py --input sp500_user.csv \
        --output sp500_returns_local.csv

The output has columns ``Date,Returns`` where Returns is the simple daily
return Close_t / Close_{t-1} - 1, matching the convention used in the paper.
Pass ``--log-returns`` to emit log returns instead.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a daily-return CSV from a user-supplied S&P 500 close series."
    )
    parser.add_argument(
        "--input", type=Path, required=True,
        help="Path to a user-provided CSV with at least Date and Close columns.",
    )
    parser.add_argument(
        "--output", type=Path, default=Path(__file__).resolve().parent / "sp500_returns_local.csv",
        help="Where to write the Date,Returns file (default: data/sp500_returns_local.csv).",
    )
    parser.add_argument(
        "--date-start", default="2010-01-01",
        help="Inclusive lower date bound (default 2010-01-01, matching the paper).",
    )
    parser.add_argument(
        "--date-end", default="2023-12-31",
        help="Inclusive upper date bound (default 2023-12-31, matching the paper).",
    )
    parser.add_argument(
        "--close-col", default="Close",
        help="Name of the close-price column in the input (default: Close).",
    )
    parser.add_argument(
        "--log-returns", action="store_true",
        help="Emit log returns log(Close_t/Close_{t-1}) instead of simple returns.",
    )
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(
            f"Input file not found: {args.input}\n"
            "Supply your own lawfully obtained S&P 500 daily close series. "
            "This repository does not redistribute S&P 500 index data."
        )

    df = pd.read_csv(args.input)
    if "Date" not in df.columns or args.close_col not in df.columns:
        raise ValueError(
            f"Expected 'Date' and '{args.close_col}' columns in {args.input}; "
            f"found {list(df.columns)}."
        )

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    df = df[(df["Date"] >= args.date_start) & (df["Date"] <= args.date_end)].copy()

    close = df[args.close_col].to_numpy(dtype=float)
    if args.log_returns:
        returns = np.diff(np.log(close))
    else:
        returns = close[1:] / close[:-1] - 1.0

    out = pd.DataFrame({"Date": df["Date"].to_numpy()[1:], "Returns": returns})
    out = out.replace([np.inf, -np.inf], np.nan).dropna(subset=["Returns"])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"Wrote {len(out)} daily returns to {args.output}")
    print(f"  date range: {out['Date'].min().date()} .. {out['Date'].max().date()}")
    print(f"  mean={out['Returns'].mean():.6e}  std={out['Returns'].std(ddof=1):.6e}")


if __name__ == "__main__":
    main()
