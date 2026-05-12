"""`os identify-vessel <photo>` — Gemma 4 multimodal on a natural image.

This is the honest multimodal showcase. Mel spectrograms are too
domain-shifted for Gemma 4 e4b to read (tested: 1/4 accuracy, strong
SHIP bias). Natural images, however, are Gemma's home turf:
vessel-on-water photographs are exactly the kind of image the model
was trained to understand.

Workflow:
  - Operator (drone pilot, patrol boat, port observer) photographs a
    suspect vessel
  - `os identify-vessel <photo>` sends to Gemma 4 multimodal
  - Gemma returns: vessel type, approximate size, visible gear,
    threat assessment, recommended cross-reference checks
  - Operator pairs this with the acoustic detection (`os detect`) and
    the intelligence brief (`os brief`) for full situational awareness

Designed to be honest: we surface Gemma's raw response without
post-processing it to look more confident than it is.
"""
from __future__ import annotations

import time
from pathlib import Path

import typer


PROMPT = """\
You are a maritime intelligence analyst for an MPA enforcement team.
You are looking at a single photograph that an officer or drone has
captured. Your job is to extract everything operationally useful.

Answer in this exact format (one section per line, leave blank if
genuinely unsure rather than guessing):

VESSEL_TYPE:        e.g. stern trawler, cargo ship, recreational fishing boat, sailboat, RHIB, unknown
APPROXIMATE_LENGTH: in metres, with uncertainty range
VISIBLE_GEAR:       fishing gear / cargo / equipment you can see (trawl door, longline, crane, nets …)
FLAG_OR_NUMBER:     any hull number, name, or visible flag (transcribe exactly)
ACTIVITY:           what the vessel appears to be doing right now
CONDITION:          sea conditions / time of day / visibility
THREAT_ASSESSMENT:  short paragraph — is this consistent with legal activity in an MPA, or
                    is the visible behaviour suspicious? Cite specific image features.
CROSS_REFERENCE:    what data sources to check next (AIS lookup of hull number, fishing
                    permit registry, etc.) to verify or escalate.

Be specific. Mention what you actually see in the image, not what you'd
expect to see. If the image is too poor to identify something, say
'unknown — image quality insufficient' rather than guessing.
"""


def identify_vessel_command(
    photo: str = typer.Argument(..., help="Path to a vessel photograph (jpg/png)."),
    model: str = typer.Option(
        "gemma4:e4b", "--model",
        help="Gemma 4 variant to use via Ollama.",
    ),
) -> None:
    """Gemma 4 multimodal vessel identification from a natural photograph.

    Honest scope: works on natural vessel-on-water photos (Gemma 4's home
    turf). Mel spectrograms are out of scope — Gemma does not read them
    reliably (tested 1/4 on hydrophone data). Use `os detect` for acoustic
    classification and this command for visual identification; pair them
    in the intelligence brief.
    """
    from .ui import console, fail, ok

    path = Path(photo)
    if not path.exists():
        fail(f"Photo not found: {photo}")
        raise typer.Exit(code=2)

    suffix = path.suffix.lower()
    if suffix not in {".jpg", ".jpeg", ".png", ".webp", ".gif"}:
        fail(f"Unsupported image format: {suffix}. Use jpg/png/webp.")
        raise typer.Exit(code=2)

    try:
        import ollama
    except Exception as e:
        fail(f"Ollama not installed: {e}")
        raise typer.Exit(code=1)

    console.rule("[bold]Ocean Sentinel · vessel identification (Gemma 4 multimodal)")
    console.print(f"  Photo: [cyan]{path}[/cyan]  ({path.stat().st_size // 1024} KB)")
    console.print(f"  Model: {model}")
    console.print()
    console.print("[dim]  Sending to Gemma 4 multimodal...[/dim]")

    t0 = time.perf_counter()
    try:
        client = ollama.Client(host="http://localhost:11434", timeout=120.0)
        response = client.chat(
            model=model,
            messages=[{
                "role": "user",
                "content": PROMPT,
                "images": [str(path)],
            }],
        )
    except Exception as e:
        fail(f"Ollama call failed: {type(e).__name__}: {e}")
        raise typer.Exit(code=1)

    elapsed = time.perf_counter() - t0
    raw = (response.get("message", {}) or {}).get("content", "").strip()

    console.print()
    console.print(raw)
    console.print()
    ok(f"Gemma 4 latency: {elapsed:.1f}s")
    console.print("[dim]  Next: cross-reference visible identifiers with AIS via `os brief` "
                  "or query the acoustic memory for similar past vessels via "
                  "`os detect <wav> --memory`.[/dim]")
