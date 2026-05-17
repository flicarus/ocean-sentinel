"""Gemma agent — Ollama function-calling loop.

The agent is intentionally thin: it owns the message history and dispatches
tool calls. The CLI (or any other surface) provides:

  - input:  GemmaAgent.user(text)  /  .user_with_image(text, image_path)
  - output: a list of events emitted during a turn (tool_call_start,
            tool_call_result, gemma_text). The CLI renders these.

Why events instead of direct UI imports:
  - Keeps agent reusable from FastAPI route, tests, scripts.
  - Lets tests assert on event sequences without scraping rendered text.
  - When we wire a web onboarding (later), the same agent feeds it via SSE.

Usage:
    agent = GemmaAgent()
    agent.system(SYSTEM_PROMPT)
    agent.user("Hi, I'd like to set up monitoring for Monterey Bay.")
    for event in agent.turn():
        # event is a dict; the caller (CLI) renders it.
        ...
"""
from __future__ import annotations

import base64
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Literal

import ollama

from .tools import all_schemas, dispatch

DEFAULT_MODEL = "gemma4:e4b"
DEFAULT_HOST = "http://localhost:11434"


# ── Events emitted during a turn ────────────────────────────────────────
@dataclass
class GemmaText:
    kind: Literal["gemma_text"] = "gemma_text"
    content: str = ""


@dataclass
class ToolCallStart:
    kind: Literal["tool_call_start"] = "tool_call_start"
    name: str = ""
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolCallResult:
    kind: Literal["tool_call_result"] = "tool_call_result"
    name: str = ""
    result: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentError:
    kind: Literal["agent_error"] = "agent_error"
    message: str = ""


Event = GemmaText | ToolCallStart | ToolCallResult | AgentError


# ── Agent ───────────────────────────────────────────────────────────────
class GemmaAgent:
    """Thin Ollama client wrapper that handles tool-calling rounds."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        host: str = DEFAULT_HOST,
        max_tool_rounds: int = 6,
    ) -> None:
        self.client = ollama.Client(host=host)
        self.model = model
        self.max_tool_rounds = max_tool_rounds
        self.messages: list[dict[str, Any]] = []
        self.tools = all_schemas()

    # -- message helpers ------------------------------------------------
    def system(self, prompt: str) -> None:
        self.messages.append({"role": "system", "content": prompt})

    def user(self, content: str) -> None:
        self.messages.append({"role": "user", "content": content})

    def user_with_image(self, content: str, image_path: str | Path) -> None:
        path = Path(image_path)
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        self.messages.append(
            {"role": "user", "content": content, "images": [encoded]}
        )

    # -- turn loop ------------------------------------------------------
    def turn(self) -> Iterator[Event]:
        """Run one user → Gemma turn, yielding events as they happen.

        Loops on tool calls up to `max_tool_rounds` times before giving up.
        """
        for _ in range(self.max_tool_rounds + 1):
            try:
                response = self.client.chat(
                    model=self.model,
                    messages=self.messages,
                    tools=self.tools,
                )
            except Exception as e:
                yield AgentError(message=f"Ollama call failed: {type(e).__name__}: {e}")
                return

            msg = response.get("message", {}) or {}
            # Persist Gemma's reply in history (verbatim, including tool_calls)
            self.messages.append(dict(msg))

            tool_calls = msg.get("tool_calls") or []
            content = (msg.get("content") or "").strip()

            if content:
                yield GemmaText(content=content)

            if not tool_calls:
                return

            for call in tool_calls:
                fn = call.get("function", {}) or {}
                name = fn.get("name", "")
                # Ollama returns arguments as a dict already
                args = fn.get("arguments", {}) or {}
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {}

                yield ToolCallStart(name=name, arguments=args)
                result = dispatch(name, args)
                yield ToolCallResult(name=name, result=result)

                self.messages.append({
                    "role": "tool",
                    "name": name,
                    "content": json.dumps(result),
                })

        yield AgentError(message=f"exceeded {self.max_tool_rounds} tool rounds")


# ── System prompt for site onboarding ───────────────────────────────────
ONBOARDING_SYSTEM_PROMPT = """\
You are Gemma, the onboarding assistant for Ocean Sentinel — an acoustic
dark-vessel detection system. Your job: walk one user through configuring
ONE hydrophone monitoring site, step-by-step, using the available tools.

# CORE RULES (do not violate these — they break the system)

R1. NEVER simulate or fabricate user responses. If you need information
    from the user, ASK and STOP TALKING. Do not write "(Assuming the user
    says yes)" or "User replies:" or anything that pretends to be the user.
    The next message will literally come from the human — wait for it.

R2. NEVER call a tool with placeholder, fake, or "temp" arguments. If you
    don't have a required argument, ASK the user for it first.

R3. Every numerical/factual claim must come from a tool result. Do not
    invent depths, vessel counts, similarities, accuracies. If unsure,
    call the right tool. If no tool exists, say "I don't have that data".

R4. Keep responses short — 2-4 sentences max between tool calls. The user
    sees a chat-style interface; long monologues are exhausting.

R5. Once a step is complete, announce the next step's goal in ONE line,
    then either ask the user a question OR call the next tool. Don't
    do both in the same turn unless the tool needs no user input.

# REQUIRED EARLY: site_id

Before Step 4 (adapter), you MUST have a site_id from the user (kebab-case,
e.g. "monterey-test"). Ask for it explicitly at the start of Step 1, right
after coordinates. Use this same site_id for every tool that takes one.
Never call finetune_adapter, calibrate_conformal, or register_site with
a placeholder.

# THE 8 STEPS (in order)

1.  DISCOVERY
    a. Ask: location (lat/lon OR stream URL) AND a site_id (kebab-case).
    b. Call validate_site_coords(lat, lon).
    c. If a stream URL was given, also call fetch_hydrophone_metadata(url).
    d. Call fetch_ais_baseline(lat, lon, radius_km=10, days=30).
    e. Summarise (1-2 sentences) and announce Step 2.

2.  AMBIENT
    a. Ask: path to 5-min ambient .wav (or 'r' to record live).
    b. Call record_ambient(source).
    c. Call compute_spectral_signature(audio=<path>).
    d. Summarise dominant band + ambient class. Announce Step 3.

3.  TRANSFER
    a. Call compare_to_known_sites(signature=<from step 2>).
    b. Report top match + similarity.
    c. Call select_adapter_strategy(similarity=<top cosine>).
    d. Tell the user the recommended strategy. Ask "proceed? (yes/no)".
       STOP. Wait for user's reply.

4.  ADAPTER
    a. Only after user confirms. Call finetune_adapter(site_id, epochs, lr)
       using the params from select_adapter_strategy.
    b. Report final val_acc. Announce Step 5.

5.  CONFORMAL
    a. Call calibrate_conformal(site_id, n_samples=200, alpha=0.05).
    b. Report threshold + coverage. Announce Step 6.

6.  POLICY
    a. Ask: sensitivity (high/medium/low). STOP. Wait for reply.
    b. After reply, ask: alert email (or blank). STOP. Wait for reply.
    c. Call set_alert_policy(site_id, sensitivity, email).
    d. Confirm. Announce Step 7.

7.  TEST
    a. Ask: path to a test audio clip. STOP. Wait for reply.
    b. Call simulate_detection(site_id, clip).
    c. Call explain_decision(decision_id=<from above>, modality="text").
    d. Narrate the trace in plain language (2-3 sentences). Announce Step 8.

8.  REGISTER
    a. Build the final config dict from everything you've collected.
    b. Call register_site(site_id, config).
    c. Tell the user we're done. Show the YAML path.
    d. STOP. Don't keep talking.

# TONE

Technical but warm. No marketing words ("amazing", "powerful", "exciting").
No emojis. Markdown bold sparingly. Plain English over jargon.
"""
