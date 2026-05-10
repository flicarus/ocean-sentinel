"""Compare two or more model checkpoints side-by-side.

Reads the JSON outputs of `eval_per_site.py` (raw CNN accuracy) and
`eval_end_to_end.py` (decision-tier outcomes) and prints a single table
with per-site deltas.

Usage:
    venv/bin/python scripts/compare_models.py \\
        --raw data/eval/per_site.json:v7 \\
              data/eval/per_site_v7_1.json:v7.1 \\
              data/eval/per_site_v7_2.json:v7.2 \\
        --e2e data/eval/end_to_end_v7_1_oracle.json:v7.1 \\
              data/eval/end_to_end_v7_2_oracle.json:v7.2

Each --raw / --e2e arg is `path:label` so the table headers stay readable.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _load(spec: str) -> tuple[str, dict]:
    """Parse path:label, load json, return (label, data)."""
    if ":" in spec:
        path_str, label = spec.rsplit(":", 1)
    else:
        path_str, label = spec, Path(spec).stem
    p = Path(path_str)
    if not p.exists():
        print(f"WARN: missing {p}", file=sys.stderr)
        return label, {}
    return label, json.loads(p.read_text())


def _print_raw_table(specs: list[tuple[str, dict]]) -> None:
    """Per-site raw CNN accuracy across models."""
    if not specs:
        return
    sites = sorted({
        site for _, d in specs for site in (d.get("per_site") or {})
    })
    if not sites:
        return
    labels = [lab for lab, _ in specs]
    print("\n=== Raw CNN per-site accuracy ===\n")
    header = f"{'site':<28}" + "".join(f"{lab:>10}" for lab in labels)
    print(header)
    print("-" * len(header))
    for site in sites:
        cells = []
        for _, d in specs:
            r = (d.get("per_site") or {}).get(site)
            cells.append(f"{r['accuracy']:>9.1%}" if r else f"{'-':>9}")
        print(f"{site:<28}" + "".join(c.rjust(10) for c in cells))
    print("-" * len(header))
    cells = []
    for _, d in specs:
        ov = d.get("overall") or {}
        acc = ov.get("accuracy")
        cells.append(f"{acc:>9.1%}" if acc is not None else f"{'-':>9}")
    print(f"{'OVERALL':<28}" + "".join(c.rjust(10) for c in cells))


def _print_e2e_table(specs: list[tuple[str, dict]]) -> None:
    """End-to-end commit_acc / abstain / wrong across models."""
    if not specs:
        return
    sites = sorted({
        site for _, d in specs for site in (d.get("per_site") or {})
    })
    if not sites:
        return
    labels = [lab for lab, _ in specs]
    print("\n=== End-to-end commit_acc per-site ===\n")
    header = f"{'site':<28}" + "".join(f"{lab:>10}" for lab in labels)
    print(header)
    print("-" * len(header))
    for site in sites:
        cells = []
        for _, d in specs:
            r = (d.get("per_site") or {}).get(site)
            ca = r.get("correct_when_committed") if r else None
            cells.append(f"{ca:>9.1%}" if ca is not None else f"{'-':>9}")
        print(f"{site:<28}" + "".join(c.rjust(10) for c in cells))
    print("-" * len(header))
    for metric, label in [
        ("correct_when_committed", "OVERALL acc"),
        ("abstain_rate", "OVERALL abstain"),
        ("wrong_rate", "OVERALL wrong"),
    ]:
        cells = []
        for _, d in specs:
            v = (d.get("overall") or {}).get(metric)
            cells.append(f"{v:>9.1%}" if v is not None else f"{'-':>9}")
        print(f"{label:<28}" + "".join(c.rjust(10) for c in cells))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--raw", nargs="*", default=[],
                   help="path:label entries for raw per-site eval JSONs")
    p.add_argument("--e2e", nargs="*", default=[],
                   help="path:label entries for end-to-end eval JSONs")
    args = p.parse_args()

    raw_specs = [_load(s) for s in args.raw]
    e2e_specs = [_load(s) for s in args.e2e]

    if raw_specs:
        _print_raw_table(raw_specs)
    if e2e_specs:
        _print_e2e_table(e2e_specs)
    if not raw_specs and not e2e_specs:
        print("Pass --raw and/or --e2e with path:label entries.")


if __name__ == "__main__":
    main()
