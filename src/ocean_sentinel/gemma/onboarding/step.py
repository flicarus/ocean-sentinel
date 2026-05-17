"""Step protocol — every onboarding step implements this contract.

A step is a pure unit of work: given the current context, the agent, and
a UI sink, drive Gemma + tools to produce a well-defined set of outputs
that get written back into the context. The orchestrator handles retries,
checkpointing, and step transitions — steps don't have to care about that.

Why a protocol (not just functions):
- We need step *metadata* (name, index, tools subset, prompt) for the UI
  and for telemetry, separately from the run logic.
- Steps need to declare their *contract* (required inputs, produced outputs)
  so the orchestrator can validate before/after each run.
- Tests can introspect steps without running them.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol

from .context import OnboardingContext


class StepStatus(str, Enum):
    OK = "ok"
    NEEDS_RETRY = "needs_retry"   # Gemma errored / got confused — try again
    USER_ABORT = "user_abort"     # User typed 'exit' / 'no'
    HARD_FAIL = "hard_fail"       # Tool call failed beyond recovery


@dataclass
class StepResult:
    status: StepStatus
    message: str = ""
    extracted: dict[str, Any] = field(default_factory=dict)


class UISink(Protocol):
    """Minimal UI surface a step needs. Concrete impl lives in cli/main.py."""

    def step_header(self, n: int, total: int, title: str, subtitle: str = "") -> None: ...
    def gemma_say(self, text: str) -> None: ...
    def ask(self, question: str, hint: str = "", default: str | None = None) -> str: ...
    def confirm(self, question: str, default_yes: bool = True) -> bool: ...
    def tool_call_static(self, name: str, args: dict[str, Any]) -> None: ...
    def ok(self, text: str) -> None: ...
    def fail(self, text: str) -> None: ...
    def warn(self, text: str) -> None: ...
    def error(self, text: str) -> None: ...


class AgentSink(Protocol):
    """Minimal agent surface a step needs. Real impl is GemmaAgent."""

    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]]

    def system(self, prompt: str) -> None: ...
    def user(self, content: str) -> None: ...
    def turn(self) -> Any: ...   # iterator of events


class Step(ABC):
    """Abstract base for an onboarding step."""

    #: 1-based step index (1..8)
    index: int = 0
    #: Short human title rendered in step header
    title: str = ""
    #: One-line subtitle ("what this step is about")
    subtitle: str = ""
    #: Names of tools (from gemma.tools.TOOLS) Gemma is allowed to call
    #: in this step. Other tools are filtered out before agent.turn().
    relevant_tools: tuple[str, ...] = ()
    #: System prompt for this step (focused, short — see prompts.py)
    system_prompt: str = ""

    @abstractmethod
    def required(self, ctx: OnboardingContext) -> list[str]:
        """Return human-readable names of ctx fields that MUST be set before
        this step starts. Empty list = step has no preconditions."""
        ...

    @abstractmethod
    def run(self, ctx: OnboardingContext, agent: AgentSink, ui: UISink) -> StepResult:
        """Drive the step. Mutate ctx with extracted state. Return a
        StepResult so the orchestrator knows what to do next."""
        ...

    # ── Helpers shared by concrete steps ───────────────────────────────
    def filtered_tools(self, all_tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Return only tool schemas whose function.name is in relevant_tools."""
        if not self.relevant_tools:
            return all_tools
        allowed = set(self.relevant_tools)
        return [t for t in all_tools if t.get("function", {}).get("name") in allowed]

    def install(self, agent: AgentSink) -> None:
        """Reset agent state for this step: replace system prompt, restrict
        tools to the relevant subset. Preserves prior message history so
        Gemma keeps full conversational context."""
        # Strip old system messages — keep user/assistant/tool history.
        agent.messages = [m for m in agent.messages if m.get("role") != "system"]
        # Insert this step's system prompt at the top.
        agent.messages.insert(0, {"role": "system", "content": self.system_prompt})
        # Restrict tool list.
        from ..tools import all_schemas
        agent.tools = self.filtered_tools(all_schemas())

    def validate_preconditions(self, ctx: OnboardingContext) -> list[str]:
        """Return a list of missing required fields, or empty if all set."""
        missing = []
        for fname in self.required(ctx):
            val = getattr(ctx, fname, None)
            if val is None or val == "" or val == []:
                missing.append(fname)
        return missing
