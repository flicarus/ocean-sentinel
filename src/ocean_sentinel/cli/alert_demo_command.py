"""`os alert-demo` — end-to-end mock-stream demo of the alert pipeline.

What it does:
  1. Replays a WAV file via MockStreamFeeder at configurable cadence.
  2. For each 60-second chunk, runs v7.6 CNN inference.
  3. Maps CNN output to decision_tier (DARK_VESSEL / CONFIRMED_VESSEL / …).
  4. Computes ThreatLevel via threat_scoring (factors MPA, sensitivity,
     CPA distance, nighttime).
  5. If threat ≥ minimum, dispatches to configured channels
     (SendGrid email + Twilio SMS + generic webhook).
  6. Streams the full event log to the console, ready for video capture.

Designed for the hackathon submission demo. Production paths exist in
`os monitor` (file-based) and the FastAPI `/api/alerts` route.
"""
from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timezone
from pathlib import Path

import typer


def alert_demo_command(
    stream: str = typer.Argument(..., help="WAV/FLAC source for the mock hydrophone stream."),
    site_id: str = typer.Option("oc01", "--site", help="Hydrophone site_id; selects MPA + thresholds."),
    phone: str | None = typer.Option(None, "--phone", help="E.164 phone number for SMS (Twilio)."),
    email: str | None = typer.Option(None, "--email", help="Email recipient (SendGrid)."),
    webhook_url: str | None = typer.Option(None, "--webhook", help="Generic webhook URL."),
    min_threat: str = typer.Option("MEDIUM", "--min-threat", help="LOW / MEDIUM / HIGH / CRITICAL."),
    realtime: bool = typer.Option(False, "--realtime", help="Sleep between chunks for real-time pace."),
    in_mpa: bool = typer.Option(True, "--in-mpa/--no-mpa", help="Site is inside a Marine Protected Area."),
    sensitivity: str = typer.Option("high", "--sensitivity", help="low / medium / high"),
    max_chunks: int = typer.Option(10, "--max-chunks", help="Hard cap on chunks (so demo terminates)."),
    # AIS context — mock values for demo. In production these come from a
    # live GFW/MarineCadastre stream watcher.
    ais_vessels: int = typer.Option(
        0, "--ais-vessels",
        help="Mock: number of AIS-broadcasting vessels in radius right now.",
    ),
    gone_dark: int = typer.Option(
        0, "--gone-dark",
        help="Mock: number of vessels that just stopped broadcasting AIS "
             "(within last 30 min). Classic illegal-fishing playbook.",
    ),
    nighttime: bool = typer.Option(
        False, "--night", help="Mock: night-time detection (escalates fishing enforcement)."
    ),
) -> None:
    """End-to-end mock-stream alert pipeline demo."""
    from .ui import console, ok, fail
    from ..adapters.mock_stream import MockStreamFeeder
    from ..services.cnn_v7_classifier import CNNV7Classifier
    from ..domain.threat_scoring import compute_threat_level, SiteContext
    from ..domain.decision_tier import (
        decide_tier, gone_dark_alone_tier, AISContext,
    )
    from ..domain.enums import AlertChannel, ThreatLevel
    from ..domain.models import (
        Alert, AudioSegment, DetectionEvent, GeoPoint, TimeWindow,
    )
    from ..adapters.alerts.dispatcher import (
        ChannelConfig, SiteAlertPolicy, dispatch_alerts,
    )
    from ..config import Settings

    settings = Settings()

    src = Path(stream)
    if not src.exists():
        fail(f"stream source not found: {stream}")
        raise typer.Exit(2)

    # Build alert policy from CLI args
    channels: list[ChannelConfig] = []
    if email:
        channels.append(ChannelConfig(AlertChannel.EMAIL, email, ThreatLevel[min_threat]))
    if phone:
        channels.append(ChannelConfig(AlertChannel.SMS, phone, ThreatLevel[min_threat]))
    if webhook_url:
        channels.append(ChannelConfig(AlertChannel.WEBHOOK, webhook_url, ThreatLevel[min_threat]))

    policy = SiteAlertPolicy(site_id=site_id, channels=channels)

    # Site context for threat scoring
    site_ctx = SiteContext(
        site_id=site_id,
        in_mpa=in_mpa,
        mpa_name="OC01 / Olympic Coast NMS" if in_mpa else None,
        sensitivity=sensitivity,
        mpa_buffer_km=5.0 if not in_mpa else None,
    )

    console.rule(f"[bold]Ocean Sentinel · alert pipeline demo")
    console.print(f"  source:       [cyan]{stream}[/cyan]")
    console.print(f"  site:         [cyan]{site_id}[/cyan]  (in_mpa={in_mpa}, sensitivity={sensitivity})")
    console.print(f"  AIS context:  vessels_in_radius={ais_vessels}, recently_gone_dark={gone_dark}, night={nighttime}")
    console.print(f"  channels:     " + (", ".join(f"{c.channel.value}→{c.recipient}" for c in channels) or "[dim]none configured[/dim]"))
    console.print(f"  min_threat:   {min_threat}")
    console.print(f"  realtime:     {realtime}")
    console.print()

    # Set up classifier
    clf = CNNV7Classifier("data/models/cnn_v7_6.pt")
    clf.set_site_thresholds("data/calibration/per_site_thresholds_v7_6.json")
    active_thr = clf._threshold_for(site_id)

    feeder = MockStreamFeeder(src, chunk_s=60.0, realtime=realtime)
    console.print(f"  source duration: {feeder.duration_s:.1f}s, expected chunks: {feeder.n_chunks}")
    console.print()

    ais_ctx = AISContext(
        ais_vessels_in_radius=ais_vessels,
        recently_gone_dark_count=gone_dark,
    )

    async def _go() -> None:
        n = 0

        # If a vessel just went dark in an MPA, fire that alert FIRST
        # (independent of any acoustic detection in the stream). This is
        # the "AIS gap as primary trigger" path the user asked for.
        ais_gap_tier = gone_dark_alone_tier(
            in_mpa=in_mpa,
            recently_gone_dark_count=gone_dark,
        )
        if ais_gap_tier is not None:
            assessment_gap = compute_threat_level(
                ais_gap_tier, site_ctx, cpa_km=None, nighttime=nighttime,
            )
            console.print(
                f"[bold magenta]AIS-gap trigger[/bold magenta]  "
                f"tier=[yellow]{ais_gap_tier}[/yellow]  "
                f"threat=[red bold]{assessment_gap.threat_level.value}[/red bold]"
            )
            console.print(f"  [dim]{assessment_gap.reasoning}[/dim]")
            if channels:
                event_gap = DetectionEvent(
                    id=f"GAP-{uuid.uuid4().hex[:8]}",
                    timestamp=datetime.now(timezone.utc),
                    location=GeoPoint(lat=48.400, lon=-124.700),
                    threat_level=assessment_gap.threat_level,
                    confidence=1.0,  # AIS-gap is a binary fact, not a prob
                    classification_reasoning=f"{ais_gap_tier}: {assessment_gap.reasoning}",
                    audio_segment=None,
                    ais_gaps=[],
                    ocean_conditions=None,
                    raw_model_output={"trigger": "ais_gap", "tier": ais_gap_tier},
                )
                alerts_gap = await dispatch_alerts(event_gap, policy, settings)
                for a in alerts_gap:
                    status_color = "green" if a.status.value == "sent" else "red"
                    console.print(
                        f"    → [{status_color}]{a.channel.value}[/] to {a.recipient}  "
                        f"[{status_color}]{a.status.value}[/]"
                        + (f"  [dim]({a.failure_reason})[/dim]" if a.failure_reason else "")
                    )

        for chunk in feeder.iter_chunks():
            n += 1
            if n > max_chunks:
                console.print(f"[dim]reached --max-chunks={max_chunks}, stopping[/dim]")
                break

            t_now = datetime.now(timezone.utc)
            pred = clf.predict(chunk.spectrogram, source_id=site_id)
            ship_prob = float(pred["probabilities"]["ship"])
            uncertainty = float(pred["uncertainty"])
            label = pred["label"]

            # Decision tier — proper domain logic factors AIS state
            tier, _sev = decide_tier(
                ship_prob=ship_prob,
                uncertainty=uncertainty,
                site_threshold=active_thr,
                ais=ais_ctx,
            )

            # Threat level
            assessment = compute_threat_level(
                tier, site_ctx,
                cpa_km=ais_ctx.nearest_vessel_cpa_km,
                nighttime=nighttime,
            )

            console.print(
                f"[bold]chunk {chunk.chunk_index}[/bold] "
                f"@ +{chunk.offset_s:.0f}s  "
                f"prob={ship_prob:.3f}  unc={uncertainty:.2f}  "
                f"tier=[yellow]{tier}[/yellow]  "
                f"threat=[{'red bold' if assessment.threat_level in (ThreatLevel.HIGH, ThreatLevel.CRITICAL) else 'cyan'}]{assessment.threat_level.value}[/]"
            )
            console.print(f"  [dim]{assessment.reasoning}[/dim]")

            if not channels:
                continue

            event = DetectionEvent(
                id=f"DEMO-{uuid.uuid4().hex[:8]}",
                timestamp=t_now,
                location=GeoPoint(lat=48.400, lon=-124.700),
                threat_level=assessment.threat_level,
                confidence=ship_prob,
                classification_reasoning=f"{tier}: {assessment.reasoning}",
                audio_segment=None,
                ais_gaps=[],
                ocean_conditions=None,
                raw_model_output={"ship_prob": ship_prob, "uncertainty": uncertainty, "tier": tier},
            )

            alerts = await dispatch_alerts(event, policy, settings)
            for a in alerts:
                status_color = "green" if a.status.value == "sent" else "red"
                console.print(
                    f"    → [{status_color}]{a.channel.value}[/] to {a.recipient}  "
                    f"[{status_color}]{a.status.value}[/]"
                    + (f"  [dim]({a.failure_reason})[/dim]" if a.failure_reason else "")
                )

    asyncio.run(_go())
    ok("alert demo complete")
