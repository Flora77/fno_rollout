#!/usr/bin/env python3
"""Aggregate sparse-experiment CSV metrics across seeds."""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path


def _parse_list(text: str) -> list[str]:
    values = [item.strip() for item in text.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected a comma-separated list")
    return values


def _as_finite_float(value: str):
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--group-by",
        type=_parse_list,
        default=_parse_list("method,coupling,mask_type,observation_rate"),
    )
    parser.add_argument("--metrics", type=_parse_list, required=True)
    args = parser.parse_args()

    rows: list[dict[str, str]] = []
    for path in args.inputs:
        with path.open("r", newline="", encoding="utf-8-sig") as handle:
            rows.extend(dict(row) for row in csv.DictReader(handle))
    if not rows:
        parser.error("no result rows were found")

    columns = set().union(*(row.keys() for row in rows))
    missing = [column for column in [*args.group_by, *args.metrics] if column not in columns]
    if missing:
        parser.error(f"missing columns: {missing}")

    groups: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row.get(column, "") for column in args.group_by)].append(row)

    output_rows: list[dict[str, object]] = []
    for group_key, group_rows in sorted(groups.items()):
        output: dict[str, object] = dict(zip(args.group_by, group_key))
        output["n_rows"] = len(group_rows)
        for metric in args.metrics:
            values = [
                value
                for value in (_as_finite_float(row.get(metric, "")) for row in group_rows)
                if value is not None
            ]
            output[f"{metric}_n"] = len(values)
            output[f"{metric}_mean"] = statistics.fmean(values) if values else ""
            output[f"{metric}_std"] = statistics.stdev(values) if len(values) >= 2 else 0.0 if values else ""
        output_rows.append(output)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(output_rows[0].keys())
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)
    print(f"summarized {len(rows)} rows into {len(output_rows)} groups: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
