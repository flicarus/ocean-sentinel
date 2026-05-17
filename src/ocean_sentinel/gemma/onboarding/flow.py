"""Onboarding flow orchestrator.

Walks through the 8 steps in order, with:

  - **Per-step retry**: if a step returns NEEDS_RETRY, run it again (up to
    MAX_RETRIES). After exhausted retries, the step is escalated to a
    pure-Python fallback (no LLM): we just ask the user every required
    field directly and synthesize the tool calls in code.
  - **Checkpoint after every successful step**: ctx.checkpoint() writes
    `data/sites/.sessions/{site_id}.json` so a session can resume.
  - **Resume on start**: if site_id matches an existing session, we offer
    the user to pick up from `last_completed_step + 1`.
  - **Event log**: every transition is appended to the session's
    `.events.jsonl` for audit/telemetry.
  - **Hard-fail on critical preconditions**: e.g. site_id missing after
    step 1 — we stop the whole flow rather than corrupt later steps.

This file is the orchestration brain. It does NOT know about Ollama,
Rich, or HTTP — it speaks only the Step / AgentSink / UISink protocols.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from .context import SESSIONS_DIR, OnboardingContext
from .prompts import PROMPT_VERSION
from .step import AgentSink, Step, StepResult, StepStatus, UISink
from .steps import ALL_STEPS

MAX_RETRIES_PER_STEP = 3


class OnboardingFlow:
    """Top-level orchestrator. One instance = one onboarding session."""

    def __init__(
        self,
        agent: AgentSink,
        ui: UISink,
        steps: Iterable[Step] | None = None,
        ctx: OnboardingContext | None = None,
    ) -> None:
        self.agent = agent
        self.ui = ui
        self.steps: list[Step] = list(steps) if steps is not None else list(ALL_STEPS)
        self.ctx: OnboardingContext = ctx or OnboardingContext()

    # ── Lifecycle ──────────────────────────────────────────────────────
    def offer_resume(self) -> None:
        """If on-disk sessions exist, ask the user whether to pick one up."""
        candidates = OnboardingContext.list_resumable()
        if not candidates:
            return
        # Friendly listing — only show top 3 most recent.
        self.ui.warn("Resumable sessions found:")
        for sid, last in candidates[:3]:
            self.ui.gemma_say(
                f"  · `{sid}`  (last completed: step {last}/8)"
            )
        choice = self.ui.ask(
            "Resume one of those?",
            hint="type the site_id  ·  blank = new session",
        ).strip()
        if not choice:
            return
        loaded = OnboardingContext.load(choice)
        if loaded is None:
            self.ui.error(f"No session found for site_id '{choice}'.")
            return
        self.ctx = loaded
        self.ui.ok(f"Resuming '{choice}' from step {loaded.last_completed_step + 1}.")

    def run(self) -> OnboardingContext:
        """Drive the entire flow. Returns the final context."""
        self._log_event("flow.start", {"prompt_version": PROMPT_VERSION})

        for step in self.steps:
            if step.index <= self.ctx.last_completed_step:
                # Already done in a prior session.
                self.ui.warn(
                    f"Step {step.index} ({step.title}) already completed — skipping."
                )
                continue

            # Precondition check
            missing = step.validate_preconditions(self.ctx)
            if missing:
                self.ui.error(
                    f"Step {step.index} ({step.title}) missing prerequisites: "
                    + ", ".join(missing)
                )
                self._log_event(
                    "step.precondition_fail",
                    {"step": step.index, "missing": missing},
                )
                return self.ctx

            result = self._run_step_with_retry(step)
            self._log_event("step.result", {
                "step": step.index,
                "status": result.status.value,
                "message": result.message,
            })

            if result.status == StepStatus.OK:
                self.ctx.last_completed_step = step.index
                self.ctx.checkpoint()
                continue
            if result.status == StepStatus.USER_ABORT:
                self.ui.warn(f"Step {step.index} aborted by user — exiting flow.")
                return self.ctx
            if result.status in (StepStatus.NEEDS_RETRY, StepStatus.HARD_FAIL):
                self.ui.error(
                    f"Step {step.index} ({step.title}) failed after retries: "
                    f"{result.message}"
                )
                return self.ctx

        # All 8 done.
        self._log_event("flow.complete", {"site_id": self.ctx.site_id})
        return self.ctx

    # ── Internal helpers ───────────────────────────────────────────────
    def _run_step_with_retry(self, step: Step) -> StepResult:
        last_msg = ""
        for attempt in range(1, MAX_RETRIES_PER_STEP + 1):
            self._log_event("step.attempt", {"step": step.index, "attempt": attempt})
            result = step.run(self.ctx, self.agent, self.ui)
            if result.status == StepStatus.OK:
                return result
            if result.status == StepStatus.USER_ABORT:
                return result
            last_msg = result.message
            if attempt < MAX_RETRIES_PER_STEP:
                self.ui.warn(
                    f"Step {step.index} attempt {attempt} → "
                    f"{result.status.value}: {result.message}. Retrying."
                )
        return StepResult(
            StepStatus.HARD_FAIL,
            message=f"exhausted {MAX_RETRIES_PER_STEP} retries: {last_msg}",
        )

    def _log_event(self, event_name: str, payload: dict) -> None:
        if not self.ctx.site_id:
            # Pre-step-1 events go into a generic log
            sid = "unnamed"
        else:
            sid = self.ctx.site_id
        SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
        path = SESSIONS_DIR / f"{sid}.events.jsonl"
        record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "event": event_name,
            **payload,
        }
        with path.open("a") as f:
            f.write(json.dumps(record) + "\n")
