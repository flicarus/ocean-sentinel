"""`os list-sites` — show all known hydrophone sites.

Lists every site that has either:
- a per-site calibrated threshold in data/calibration/per_site_thresholds_v7_6.json
- a fine-tuned adapter checkpoint in data/sites/<id>/adapter.pt
- a YAML config in data/sites/<id>/config.yaml

Per row we print: id, threshold (or 0.5 default), adapter status,
config status, and the eval accuracy for that site when available.

Useful for:
- "what sites can I pass to `os detect --site …`?"
- triage: an alert came in tagged with site X — is X really onboarded?
- compliance: list all hydrophones the model has been calibrated for.
"""
from __future__ import annotations

import json
from pathlib import Path

import typer


def list_sites_command(
    show_all: bool = typer.Option(
        False, "--all",
        help="Include sites with no threshold and no adapter (rarely useful).",
    ),
) -> None:
    """Enumerate hydrophone sites known to the system."""
    from .ui import console

    console.rule("[bold]Ocean Sentinel · known hydrophone sites")

    # Sources of truth
    thresholds_path = Path("data/calibration/per_site_thresholds_v7_6.json")
    thresholds: dict[str, float] = {}
    if thresholds_path.exists():
        try:
            doc = json.loads(thresholds_path.read_text())
            thresholds = doc.get("per_site_thresholds", {}) or {}
        except Exception:
            pass

    eval_path = Path("data/eval/per_site_v7_6.json")
    eval_per_site: dict[str, dict] = {}
    if eval_path.exists():
        try:
            d = json.loads(eval_path.read_text())
            eval_per_site = d.get("per_site", {}) or {}
        except Exception:
            pass

    sites_dir = Path("data/sites")
    adapter_sites: set[str] = set()
    config_sites: set[str] = set()
    if sites_dir.exists():
        for child in sites_dir.iterdir():
            if (child / "adapter.pt").exists():
                adapter_sites.add(child.name)
            if (child / "config.yaml").exists():
                config_sites.add(child.name)

    # Union of all known site identifiers, normalising the ais-correlated
    # prefix so 'point-robinson' and 'ais-correlated-point-robinson' don't
    # show up twice. Keep the longer form in the lookup since that's what
    # eval_per_site uses too.
    all_sites: set[str] = set()
    all_sites.update(thresholds.keys())
    all_sites.update(adapter_sites)
    all_sites.update(config_sites)
    if show_all:
        all_sites.update(eval_per_site.keys())

    if not all_sites:
        console.print("[yellow]  No sites found.[/yellow]")
        console.print("  → Run `os onboard` to add a hydrophone, or train v7.6 first.")
        return

    # Column widths
    name_w = max(len(s) for s in all_sites)
    name_w = max(name_w, 28)

    console.print(
        f"  [bold]{'site':<{name_w}}  {'threshold':>9}  {'adapter':>7}  "
        f"{'config':>6}  {'eval_acc':>8}  {'n':>5}[/bold]"
    )
    console.print(f"  [dim]{'-' * (name_w + 50)}[/dim]")

    rows = []
    for site in sorted(all_sites):
        thr = thresholds.get(site)
        # Try the alternate-name conventions for threshold lookup too
        if thr is None and not site.startswith("ais-correlated-"):
            thr = thresholds.get(f"ais-correlated-{site}")
        thr_str = f"{thr:.2f}" if thr is not None else "0.50 (def)"

        has_adapter = "✓" if site in adapter_sites else " "
        has_config = "✓" if site in config_sites else " "

        eval_row = eval_per_site.get(site)
        if not eval_row:
            # Try alternates
            eval_row = eval_per_site.get(site.replace("ais-correlated-", ""))
        if eval_row:
            acc_str = f"{eval_row.get('accuracy', 0) * 100:.1f}%"
            n_str = str(eval_row.get("n", "?"))
        else:
            acc_str = "  -  "
            n_str = "-"

        rows.append((site, thr_str, has_adapter, has_config, acc_str, n_str))

    for site, thr_str, has_adapter, has_config, acc_str, n_str in rows:
        console.print(
            f"  {site:<{name_w}}  {thr_str:>9}  {has_adapter:>7}  {has_config:>6}  "
            f"{acc_str:>8}  {n_str:>5}"
        )

    console.print()
    console.print(f"  [dim]Total sites: {len(rows)}  ·  "
                  f"with calibrated threshold: {len([r for r in rows if 'def' not in r[1]])}  ·  "
                  f"with adapter: {len(adapter_sites)}[/dim]")
    console.print()
    console.print("[dim]  Use the site id as `--site` on `os detect` / `os monitor` / `os test`.[/dim]")
