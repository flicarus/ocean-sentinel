"""Visual primitives for the Ocean Sentinel CLI.

Design goals: feel like Claude Code / Vercel CLI / Stripe CLI — restrained
palette, generous whitespace, clean typography, deliberate use of cards.

Components:
- show_banner()              top-of-session brand block
- step_header(n, total, ..)  progress dots + step title
- gemma_say(text)            chat-bubble narration (with thinking spinner)
- ask(question, hint?)       two-line prompt (question + hint + cursor)
- tool_call(name, args)      ctx mgr: card with args + spinner, caller adds result
- tool_call_static(...)      same card without spinner, for streaming results
- ok(text) / fail(text)      single-line result lines
- table(rows, title)         labeled inline table card
- site_registered(...)       big completion card
- warn(text) / error(text)   single-line status messages
"""
from __future__ import annotations

import sys
import time
from contextlib import contextmanager
from typing import Any, Iterator

from pyfiglet import Figlet
from rich.align import Align
from rich.box import HEAVY, ROUNDED
from rich.console import Console, Group
from rich.padding import Padding
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

# ── Palette ─────────────────────────────────────────────────────────────
TEAL     = "#00D4C8"
TEAL_DIM = "#1F8A85"
RED      = "#EF4444"
AMBER    = "#F97316"
GREEN    = "#22C55E"
DIM      = "grey50"
GHOST    = "grey39"
SOFT     = "grey70"

console = Console(highlight=False, soft_wrap=False)


# ── Banner ──────────────────────────────────────────────────────────────
def _render_logo() -> str:
    """SOFAR AI in ANSI Shadow — pre-rendered each session, cached after."""
    return Figlet(font="ansi_shadow", width=120).renderText("SOFAR AI")


def show_banner() -> None:
    """Top-of-session brand banner. Once per session.

    Layout:
        [BIG SOFAR AI in ANSI Shadow]
        Ocean Sentinel · acoustic dark-vessel detection · v0.7.4
    """
    console.print()

    # Big wordmark
    logo = _render_logo()
    for line in logo.splitlines():
        if line.strip():
            console.print(Padding(Text(line, style=f"bold {TEAL}"), (0, 2)))

    # Tagline
    tagline = Text()
    tagline.append("Ocean Sentinel", style="bold white")
    tagline.append("  ·  ", style=GHOST)
    tagline.append("acoustic dark-vessel detection", style=SOFT)
    tagline.append("  ·  ", style=GHOST)
    tagline.append("v0.7.4", style=DIM)
    console.print()
    console.print(Padding(tagline, (0, 4)))
    console.print()


# ── Step header ─────────────────────────────────────────────────────────
def step_header(n: int, total: int, title: str, subtitle: str = "") -> None:
    """`●●●○○○○○  Step 3 of 8   Transfer learning · find nearest sites`"""
    dots = Text()
    dots.append("●" * n, style=TEAL)
    dots.append("○" * (total - n), style=GHOST)

    line = Text()
    line.append(dots)
    line.append(f"  Step {n} of {total}   ", style=DIM)
    line.append(title, style="bold white")
    if subtitle:
        line.append(f"  ·  ", style=GHOST)
        line.append(subtitle, style=DIM)

    console.print()
    console.print(Padding(line, (0, 2)))
    console.print(Padding(Text("─" * 72, style=GHOST), (0, 2)))
    console.print()


# ── Gemma chat bubble ───────────────────────────────────────────────────
def gemma_say(text: str, *, thinking_ms: int = 500) -> None:
    """Render a chat-bubble for Gemma, with a brief 'thinking' spinner first."""
    if thinking_ms > 0 and sys.stdout.isatty():
        with console.status(
            Text("Gemma is thinking…", style=DIM),
            spinner="dots",
            spinner_style=TEAL_DIM,
        ):
            time.sleep(thinking_ms / 1000)

    label = Text("Gemma", style=f"bold {TEAL}")
    bubble = Text(text, style="white")

    body = Group(
        label,
        bubble,
    )

    panel = Panel(
        body,
        box=ROUNDED,
        border_style=TEAL_DIM,
        padding=(0, 2),
        title_align="left",
    )
    console.print(Padding(panel, (0, 2)))
    console.print()


# ── User prompt ─────────────────────────────────────────────────────────
def ask(question: str, hint: str = "", *, default: str | None = None) -> str:
    """Two-line prompt:  question (white)  / hint (dim)  / `>` cursor."""
    q = Text()
    q.append("  ▸ ", style=f"bold {GREEN}")
    q.append(question, style="white")
    console.print(q)
    if hint:
        console.print(Text(f"    {hint}", style=DIM))
    if default:
        console.print(Text(f"    default: {default}", style=DIM))
    answer = console.input(f"    [bold {GREEN}]›[/] ").strip()
    console.print()  # always a clean break after input
    return answer or (default or "")


def confirm(question: str, *, default_yes: bool = True) -> bool:
    hint = "[Y/n]" if default_yes else "[y/N]"
    a = ask(question, hint=hint).lower()
    if not a:
        return default_yes
    return a in {"y", "yes"}


# ── Tool calls ──────────────────────────────────────────────────────────
def _format_args_block(args: dict[str, Any]) -> Text:
    """Render args as a multi-line block:  `  key = value`."""
    block = Text()
    if not args:
        return block
    keylen = max(len(k) for k in args.keys())
    for i, (k, v) in enumerate(args.items()):
        if isinstance(v, str):
            vs = f'"{v}"' if len(v) <= 48 else f'"{v[:45]}…"'
        else:
            # repr handles float precision, ints, bools, None correctly
            vs = repr(v)
        block.append(f"    {k.ljust(keylen)} = ", style=DIM)
        block.append(vs, style=SOFT)
        if i < len(args) - 1:
            block.append("\n")
    return block


def _tool_card(name: str, args: dict[str, Any]) -> Group:
    head = Text()
    head.append("  ⎯ ", style=DIM)
    head.append(name, style=f"bold {TEAL}")
    return Group(head, _format_args_block(args))


@contextmanager
def tool_call(name: str, args: dict[str, Any] | None = None) -> Iterator[None]:
    """Show tool-call card; spinner stays on screen during the call."""
    args = args or {}
    console.print(_tool_card(name, args))
    with console.status(
        Text(f"  running {name}…", style=DIM),
        spinner="dots",
        spinner_style=TEAL_DIM,
    ):
        yield


def tool_call_static(name: str, args: dict[str, Any] | None = None) -> None:
    """Same card without spinner — for tools whose progress streams below
    (e.g. training epochs)."""
    console.print(_tool_card(name, args or {}))


def ok(text: str) -> None:
    line = Text()
    line.append("    ✓  ", style=GREEN)
    line.append(text, style="white")
    console.print(line)
    console.print()


def fail(text: str) -> None:
    line = Text()
    line.append("    ✗  ", style=RED)
    line.append(text, style="white")
    console.print(line)
    console.print()


# ── Result tables ───────────────────────────────────────────────────────
def table(rows: list[tuple[str, str]], title: str = "") -> None:
    """Inline labeled table card. Use after a tool_call for structured results."""
    t = Table(show_header=False, box=None, padding=(0, 2))
    t.add_column(style=DIM, no_wrap=True, width=18)
    t.add_column(style="white")
    for k, v in rows:
        t.add_row(k, v)

    panel = Panel(
        t,
        title=Text(f" {title} ", style=f"{DIM}") if title else None,
        title_align="left",
        border_style=GHOST,
        box=ROUNDED,
        padding=(0, 1),
    )
    console.print(Padding(panel, (0, 4)))
    console.print()


# ── Completion ──────────────────────────────────────────────────────────
def site_registered(site_id: str, config: dict[str, str]) -> None:
    """Big completion card with site config table inside."""
    headline = Text()
    headline.append("✓  ", style=f"bold {GREEN}")
    headline.append("Site registered: ", style="white")
    headline.append(site_id, style=f"bold {TEAL}")

    cfg = Table(show_header=False, box=None, padding=(0, 2))
    cfg.add_column(style=DIM, no_wrap=True, width=14)
    cfg.add_column(style="white")
    for k, v in config.items():
        cfg.add_row(k, v)

    body = Group(
        Align.left(headline),
        Text(""),
        cfg,
    )

    panel = Panel(
        body,
        box=HEAVY,
        border_style=TEAL,
        padding=(1, 3),
    )
    console.print(Padding(panel, (1, 2)))

    next_steps = Text()
    next_steps.append("  Next:\n", style=DIM)
    next_steps.append(f"    os monitor {site_id}", style=f"bold {TEAL}")
    next_steps.append("       start live monitoring\n", style=DIM)
    next_steps.append(f"    os explain DECISION_ID", style=f"bold {TEAL}")
    next_steps.append("   ask Gemma about a past detection\n", style=DIM)
    console.print(next_steps)


# ── Status messages ─────────────────────────────────────────────────────
def warn(msg: str) -> None:
    line = Text()
    line.append("  ⚠  ", style=AMBER)
    line.append(msg, style="white")
    console.print(line)


def error(msg: str) -> None:
    line = Text()
    line.append("  ✗  ", style=f"bold {RED}")
    line.append(msg, style="white")
    console.print(line)
