# S&P 500 data

This repository does not redistribute S&P 500 index levels or derived return
series. The empirical calibration script expects a user-supplied CSV containing
daily dates and close levels obtained from a lawful data source. The included
schema example is synthetic and is provided only to document the expected file
format.

The paper's reported S&P 500 calibration can be reproduced by placing a lawfully
obtained daily close series at `supplement/data/sp500_user.csv` and running:

```bash
python supplement/data/prepare_sp500_returns.py \
    --input supplement/data/sp500_user.csv \
    --output supplement/data/sp500_returns_local.csv

python supplement/experiments/hjb/run_sp500_mle.py \
    --returns supplement/data/sp500_returns_local.csv
```

For exact auditability without redistributing third-party market data, the
fitted MLE parameters used in the paper are stored in
`supplement/data/mle_parameters.json`.

## Files in this directory

| File | Contents |
|---|---|
| `README.md` | this file |
| `prepare_sp500_returns.py` | converts a user-supplied `Date,Close` CSV into a `Date,Returns` file |
| `sp500_schema_example.csv` | **synthetic** rows only — documents the expected input format, not real data |
| `mle_parameters.json` | the fitted VG parameters and standard errors reported in the paper |

## Expected input format

A CSV with at least `Date` and `Close` columns over the 2010-2023 window:

```
Date,Close
2010-01-04,1132.99
2010-01-05,1136.52
...
```

`prepare_sp500_returns.py` computes simple daily returns
`Close_t / Close_{t-1} - 1` (pass `--log-returns` for log returns), filters to
the paper's 2010-01-01..2023-12-31 window, and writes `Date,Returns`.
