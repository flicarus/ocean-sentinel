"""Onboarding context — typed accumulating state across the 8 steps.

Each step adds fields to the context as it completes. The context is
JSON-serialized to ``data/sites/.sessions/{site_id}.json`` after every
successful step, so a session can be resumed after Ctrl-C, crash, or
network hiccup.

The context is the *single source of truth* the orchestrator and steps
read from / write to. Steps never share data via globals or by re-asking
the user — if a value is missing, the step must explicitly require it
and fail clearly if it's still missing after a retry.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SESSIONS_DIR = Path("data/sites/.sessions")


@dataclass
class OnboardingContext:
    """Single source of truth for an in-progress site onboarding session."""

    # Required identity (set in step 1)
    site_id: str = ""
    lat: float | None = None
    lon: float | None = None
    stream_url: str | None = None

    # Step 1 outputs
    depth_m: int | None = None
    nearest_mpa: str | None = None
    ais_traffic_class: str | None = None
    ais_avg_vessels_per_day: int | None = None
    ais_shipping_lane_km: float | None = None
    hydrophone_network: str | None = None
    hydrophone_sample_rate_hz: int | None = None

    # Step 2 outputs
    ambient_source: str | None = None
    spectral_signature: list[float] = field(default_factory=list)
    ambient_class: str | None = None
    dominant_band_hz: str | None = None

    # Step 3 outputs
    nearest_known_site_id: str | None = None
    nearest_known_site_similarity: float | None = None
    adapter_strategy: str | None = None
    adapter_epochs: int | None = None
    adapter_lr: float | None = None
    user_confirmed_adapter: bool = False

    # Step 4 outputs
    adapter_val_acc: float | None = None
    adapter_checkpoint: str | None = None

    # Step 5 outputs
    conformal_threshold_p: float | None = None
    conformal_coverage: float | None = None
    expected_fa_per_hour: float | None = None

    # Step 6 outputs
    sensitivity: str | None = None
    alert_email: str | None = None
    threshold_adjust: float | None = None

    # Step 7 outputs
    test_decision_id: str | None = None
    test_decision_tier: str | None = None
    test_decision_severity: str | None = None
    test_explain_summary: str | None = None

    # Step 8 outputs
    registered_yaml_path: str | None = None

    # Bookkeeping
    started_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    last_completed_step: int = 0   # 0 = nothing done yet, 8 = fully registered
    notes: list[str] = field(default_factory=list)

    # ── Persistence ────────────────────────────────────────────────────
    def session_path(self) -> Path:
        sid = self.site_id or "unnamed"
        return SESSIONS_DIR / f"{sid}.json"

    def checkpoint(self) -> None:
        """Write the current context to disk. Called after each successful step."""
        SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
        self.session_path().write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, site_id: str) -> "OnboardingContext | None":
        """Try to resume a session. Returns None if no checkpoint exists."""
        path = SESSIONS_DIR / f"{site_id}.json"
        if not path.exists():
            return None
        data = json.loads(path.read_text())
        valid_keys = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in valid_keys})

    @classmethod
    def list_resumable(cls) -> list[tuple[str, int]]:
        """Return [(site_id, last_completed_step)] for all on-disk sessions
        that aren't fully registered yet."""
        if not SESSIONS_DIR.exists():
            return []
        out = []
        for p in SESSIONS_DIR.glob("*.json"):
            try:
                data = json.loads(p.read_text())
                if int(data.get("last_completed_step", 0)) < 8:
                    out.append((data.get("site_id", p.stem), int(data["last_completed_step"])))
            except (json.JSONDecodeError, KeyError, ValueError):
                continue
        return sorted(out)

    # ── Convenience views ──────────────────────────────────────────────
    def has_coords(self) -> bool:
        return self.lat is not None and self.lon is not None

    def to_yaml_config(self) -> dict[str, Any]:
        """Build the config dict that will be persisted to data/sites/{site_id}.yaml.
        Excludes session bookkeeping and any None values."""
        skip = {"started_at", "last_completed_step", "notes", "spectral_signature"}
        out: dict[str, Any] = {}
        for f in fields(self):
            if f.name in skip:
                continue
            val = getattr(self, f.name)
            if val is not None and val != "" and val != []:
                out[f.name] = val
        return out
