#!/usr/bin/env python
"""Prepare fixed S&P 500 returns and calibrate Variance Gamma by MLE."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from levy_flows.calibration.vg_mle import calibrate_vg_mle


ROOT = Path(__file__).resolve().parents[2]
# This repository does NOT redistribute S&P 500 index levels or derived returns.
# Build the local returns file first with data/prepare_sp500_returns.py from your
# own lawfully obtained daily close series. See data/README.md.
DEFAULT_RETURNS = ROOT / "data" / "sp500_returns_local.csv"
DEFAULT_OUTPUT = ROOT / "results" / "hjb_sp500_mle"


def load_returns(returns_path: Path) -> pd.DataFrame:
    """Load the user-prepared 2010-2023 daily-return CSV (Date,Returns)."""
    if not returns_path.exists():
        raise FileNotFoundError(
            f"Returns file not found: {returns_path}\n"
            "This repository does not ship S&P 500 data. Prepare it locally:\n"
            "  python data/prepare_sp500_returns.py --input <your_close_series.csv> "
            f"--output {returns_path}\n"
            "See data/README.md. The fitted parameters used in the paper are in "
            "data/mle_parameters.json."
        )

    df = pd.read_csv(returns_path)
    if "Date" not in df.columns or "Returns" not in df.columns:
        raise ValueError(f"Expected Date and Returns columns in {returns_path}")

    df["Date"] = pd.to_datetime(df["Date"])
    df = df[(df["Date"] >= "2010-01-01") & (df["Date"] <= "2023-12-31")].copy()
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["Returns"])
    df = df[["Date", "Returns"]]
    df = df.sort_values("Date").reset_index(drop=True)
    return df


def summarize_returns(returns: np.ndarray) -> dict:
    """Compute return sample moments."""
    return {
        "n": int(len(returns)),
        "mean_daily": float(np.mean(returns)),
        "std_daily": float(np.std(returns, ddof=1)),
        "mean_annualized": float(np.mean(returns) * 252),
        "vol_annualized": float(np.std(returns, ddof=1) * np.sqrt(252)),
        "skewness": float(stats.skew(returns, bias=False)),
        "excess_kurtosis": float(stats.kurtosis(returns, fisher=True, bias=False)),
        "min": float(np.min(returns)),
        "max": float(np.max(returns)),
    }


def json_ready(value):
    """Convert numpy arrays/scalars to JSON-compatible structures."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {k: json_ready(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_ready(v) for v in value]
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description="Run VG MLE on user-prepared S&P 500 returns.")
    parser.add_argument(
        "--returns", type=Path, default=DEFAULT_RETURNS,
        help="Path to the Date,Returns CSV produced by data/prepare_sp500_returns.py.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    df = load_returns(args.returns)
    returns = df["Returns"].to_numpy(dtype=float)
    summary = summarize_returns(returns)

    mle = calibrate_vg_mle(
        returns,
        init_params={
            "mu": summary["mean_daily"],
            "sigma": max(summary["std_daily"] * 0.8, 1e-4),
            "theta": -0.1 * summary["std_daily"] if summary["skewness"] < 0 else 0.0,
            "nu": 0.3,
        },
        compute_standard_errors=True,
    )

    result = {
        "returns_file": str(args.returns),
        "date_start": df["Date"].min().strftime("%Y-%m-%d"),
        "date_end": df["Date"].max().strftime("%Y-%m-%d"),
        "sample_moments": summary,
        "vg_mle": mle,
    }

    args.output.mkdir(parents=True, exist_ok=True)
    with open(args.output / "results.json", "w") as f:
        json.dump(json_ready(result), f, indent=2)

    table = pd.DataFrame(
        [
            {
                "parameter": name,
                "estimate": mle[name],
                "std_error": mle.get("standard_errors", {})
                .get("se_natural", {})
                .get(name, np.nan),
            }
            for name in ["mu", "sigma", "theta", "nu"]
        ]
    )
    table.to_csv(args.output / "vg_mle_parameters.csv", index=False)

    print(json.dumps(json_ready(result), indent=2))


if __name__ == "__main__":
    main()
