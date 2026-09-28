# Results

This public release intentionally does not redistribute generated result files.
Training runs, portfolio weights, logs, and evaluation tables are written here
or below experiment-specific `artifacts/` directories at runtime and are
ignored by Git.

The authoritative reported numbers are the tables in the accepted paper. In
particular, the paper's main RADAR row is:

| Sharpe | Sortino | Calmar | CumRet | AnnRet | MaxDD | Vol |
|---:|---:|---:|---:|---:|---:|---:|
| 1.062 | 1.352 | 0.924 | 2.502 | 0.285 | 0.308 | 0.271 |

`MaxDD` is reported as a positive magnitude, matching the paper.
