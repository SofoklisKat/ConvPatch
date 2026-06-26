#!/usr/bin/env python3
"""Print a summary table from runs/*/log.csv and summary.json."""

from __future__ import annotations

import csv
import json
from pathlib import Path

RUNS = Path("runs")


def best_top1(log_csv: Path) -> float | None:
    if not log_csv.exists():
        return None
    best = 0.0
    with open(log_csv) as f:
        for row in csv.DictReader(f):
            best = max(best, float(row["top1"]))
    return best


def main() -> None:
    rows = []
    for d in sorted(RUNS.glob("*_seed*")):
        log = d / "log.csv"
        summary = d / "summary.json"
        top1 = best_top1(log)
        epochs = sum(1 for _ in open(log)) - 1 if log.exists() else 0
        params = tokens = None
        if summary.exists():
            s = json.loads(summary.read_text())
            params = s.get("params_M")
            tokens = s.get("tokens")
        rows.append((d.name, epochs, top1, params, tokens))

    if not rows:
        print("No runs found in runs/")
        return

    print(f"{'run':<40} {'epochs':>6} {'best_top1':>10} {'params_M':>8} {'tokens':>6}")
    print("-" * 76)
    for name, ep, t1, pm, tok in rows:
        t1s = f"{t1:.2f}" if t1 is not None else "—"
        pms = f"{pm:.2f}" if pm is not None else "—"
        toks = str(tok) if tok is not None else "—"
        print(f"{name:<40} {ep:>6} {t1s:>10} {pms:>8} {toks:>6}")


if __name__ == "__main__":
    main()
