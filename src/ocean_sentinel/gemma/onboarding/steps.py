"""Concrete onboarding step implementations (8 of them).

All 8 steps follow the same pattern:

    1. Render header.
    2. Install per-step prompt + restrict tools (Step.install).
    3. (If needed) ask Python-driven prompts for critical fields,
       so Gemma never has to parse free-form user input for ids/numbers.
    4. Push a structured user message to Gemma (e.g. "User provided
       lat=X, lon=Y. Please validate and pull AIS context.").
    5. Iterate agent.turn(), render events to UI, extract tool results
       into the context.
    6. Validate that required outputs landed in ctx.
    7. Return StepResult(OK / NEEDS_RETRY / HARD_FAIL).

This hybrid design keeps Gemma in charge of natural-language narration
and tool dispatch, while Python keeps the flow on rails. It's the same
pattern Cursor / LangGraph / Claude Code use for production agents.
"""
from __future__ import annotations

import re
from typing import Any

from ..agent import AgentError, GemmaText, ToolCallResult, ToolCallStart
from .context import OnboardingContext
from .prompts import (
    STEP_1_DISCOVERY,
    STEP_2_AMBIENT,
    STEP_3_TRANSFER,
    STEP_4_ADAPTER,
    STEP_5_CONFORMAL,
    STEP_6_POLICY,
    STEP_7_TEST,
    STEP_8_REGISTER,
)
from .step import AgentSink, Step, StepResult, StepStatus, UISink


# ── Shared event rendering ──────────────────────────────────────────────
def render_event(ev: Any, ui: UISink) -> None:
    """Translate a GemmaAgent event into UI calls."""
    if isinstance(ev, GemmaText):
        if ev.content.strip():
            ui.gemma_say(ev.content)
    elif isinstance(ev, ToolCallStart):
        ui.tool_call_static(ev.name, ev.arguments)
    elif isinstance(ev, ToolCallResult):
        if ev.result.get("ok"):
            ui.ok(ev.result.get("summary", "✓"))
        else:
            ui.fail(ev.result.get("error", "tool failed"))
    elif isinstance(ev, AgentError):
        ui.error(ev.message)


def _drive_turn(agent: AgentSink, ui: UISink,
                ctx: OnboardingContext, extractor) -> bool:
    """Run one agent.turn(), render events, extract tool data into ctx.
    Returns True if at least one tool ran successfully or Gemma said
    something; False if turn was empty / errored hard."""
    saw_anything = False
    for ev in agent.turn():
        render_event(ev, ui)
        if isinstance(ev, (ToolCallStart, ToolCallResult)):
            extractor(ev, ctx)
            saw_anything = True
        elif isinstance(ev, GemmaText):
            saw_anything = True
        elif isinstance(ev, AgentError):
            return False
    return saw_anything


# ── Step 1: Discovery ───────────────────────────────────────────────────
_COORD_RE = re.compile(r"(-?\d{1,3}\.\d+)\s*[,\s]\s*(-?\d{1,3}\.\d+)")


def _parse_coords(text: str) -> tuple[float, float] | None:
    m = _COORD_RE.search(text)
    if not m:
        return None
    try:
        lat, lon = float(m.group(1)), float(m.group(2))
    except ValueError:
        return None
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        return None
    return lat, lon


_KEBAB_RE = re.compile(r"^[a-z][a-z0-9-]{1,40}$")


class Step1Discovery(Step):
    index = 1
    title = "Site discovery"
    subtitle = "where is your hydrophone?"
    relevant_tools = ("validate_site_coords", "fetch_ais_baseline", "fetch_hydrophone_metadata")
    system_prompt = STEP_1_DISCOVERY

    def required(self, ctx: OnboardingContext) -> list[str]:
        return []

    def run(self, ctx: OnboardingContext, agent: AgentSink, ui: UISink) -> StepResult:
        ui.step_header(self.index, 8, self.title, self.subtitle)
        self.install(agent)

        # 1. Python collects the two critical values.
        coord_in = ui.ask("Coordinates", hint="lat, lon  e.g.  36.7128, -121.9023")
        coords = _parse_coords(coord_in)
        if coords is None:
            ui.error("Couldn't parse coordinates — expected `lat, lon`.")
            return StepResult(StepStatus.NEEDS_RETRY)
        ctx.lat, ctx.lon = coords

        site_id = ui.ask("Site ID", hint="kebab-case, e.g. monterey-test")
        if not _KEBAB_RE.match(site_id):
            ui.error("site_id must be kebab-case: lowercase letters, digits, hyphens only.")
            return StepResult(StepStatus.NEEDS_RETRY)
        ctx.site_id = site_id

        # 2. Hand structured data to Gemma — she calls the tools.
        agent.user(
            f"The user provided lat={ctx.lat}, lon={ctx.lon}, site_id={ctx.site_id}. "
            f"Please validate the coordinates and pull the AIS baseline."
        )

        if not _drive_turn(agent, ui, ctx, _extract_step1):
            return StepResult(StepStatus.NEEDS_RETRY, message="empty/failed turn")

        if ctx.depth_m is None or ctx.ais_traffic_class is None:
            return StepResult(StepStatus.NEEDS_RETRY,
                              message="validation or AIS tool didn't return expected fields")
        return StepResult(StepStatus.OK)


def _extract_step1(ev: Any, ctx: OnboardingContext) -> None:
    if not isinstance(ev, ToolCallResult) or not ev.result.get("ok"):
        return
    r = ev.result
    if ev.name == "validate_site_coords":
        ctx.depth_m = r.get("depth_m")
        ctx.nearest_mpa = r.get("nearest_mpa")
    elif ev.name == "fetch_ais_baseline":
        ctx.ais_traffic_class = r.get("traffic_class")
        ctx.ais_avg_vessels_per_day = r.get("avg_vessels_per_day")
        ctx.ais_shipping_lane_km = r.get("shipping_lane_km")
    elif ev.name == "fetch_hydrophone_metadata":
        ctx.hydrophone_network = r.get("network")
        ctx.hydrophone_sample_rate_hz = r.get("sample_rate_hz")


# ── Step 2: Ambient baseline ────────────────────────────────────────────
class Step2Ambient(Step):
    index = 2
    title = "Acoustic baseline"
    subtitle = "fingerprint the site"
    relevant_tools = ("record_ambient", "compute_spectral_signature")
    system_prompt = STEP_2_AMBIENT

    def required(self, ctx: OnboardingContext) -> list[str]:
        return ["site_id"]

    def run(self, ctx: OnboardingContext, agent: AgentSink, ui: UISink) -> StepResult:
        ui.step_header(self.index, 8, self.title, self.subtitle)
        self.install(agent)

        source = ui.ask("Audio source",
                        hint="path to .wav  ·  or  'r' to record now")
        if not source:
            return StepResult(StepStatus.NEEDS_RETRY, message="no audio source")
        ctx.ambient_source = source

        agent.user(
            f"The user provided ambient audio at: {source}. "
            f"Please load it and compute the spectral signature."
        )
        if not _drive_turn(agent, ui, ctx, _extract_step2):
            return StepResult(StepStatus.NEEDS_RETRY)

        if not ctx.spectral_signature:
            return StepResult(StepStatus.NEEDS_RETRY,
                              message="spectral signature not computed")
        return StepResult(StepStatus.OK)


def _extract_step2(ev: Any, ctx: OnboardingContext) -> None:
    if not isinstance(ev, ToolCallResult) or not ev.result.get("ok"):
        return
    r = ev.result
    if ev.name == "compute_spectral_signature":
        ctx.spectral_signature = r.get("signature", []) or []
        ctx.ambient_class = r.get("ambient_class")
        ctx.dominant_band_hz = r.get("dominant_band_hz")


# ── Step 3: Transfer learning ───────────────────────────────────────────
class Step3Transfer(Step):
    index = 3
    title = "Transfer learning"
    subtitle = "find the nearest known site"
    relevant_tools = ("compare_to_known_sites", "select_adapter_strategy")
    system_prompt = STEP_3_TRANSFER

    def required(self, ctx: OnboardingContext) -> list[str]:
        return ["spectral_signature"]

    def run(self, ctx: OnboardingContext, agent: AgentSink, ui: UISink) -> StepResult:
        ui.step_header(self.index, 8, self.title, self.subtitle)
        self.install(agent)

        agent.user(
            "The user's spectral signature is in your context. Please call "
            "compare_to_known_sites with that signature, then call "
            "select_adapter_strategy with the top similarity. Then explain "
            "the recommendation in 1-2 sentences."
        )
        if not _drive_turn(agent, ui, ctx, _extract_step3):
            return StepResult(StepStatus.NEEDS_RETRY)

        if ctx.adapter_strategy is None:
            return StepResult(StepStatus.NEEDS_RETRY,
                              message="adapter_strategy not chosen")

        # Python drives the confirmation — never trust Gemma to wait.
        proceed = ui.confirm(
            f"Proceed with strategy '{ctx.adapter_strategy}' "
            f"({ctx.adapter_epochs} epochs)?",
            default_yes=True,
        )
        if not proceed:
            return StepResult(StepStatus.USER_ABORT, message="user declined adapter")
        ctx.user_confirmed_adapter = True
        return StepResult(StepStatus.OK)


def _extract_step3(ev: Any, ctx: OnboardingContext) -> None:
    if not isinstance(ev, ToolCallResult) or not ev.result.get("ok"):
        return
    r = ev.result
    if ev.name == "compare_to_known_sites":
        ranked = r.get("ranked", []) or []
        if ranked:
            top = ranked[0]
            ctx.nearest_known_site_id = top.get("id")
            ctx.nearest_known_site_similarity = top.get("cosine_sim")
    elif ev.name == "select_adapter_strategy":
        ctx.adapter_strategy = r.get("strategy")
        ctx.adapter_epochs = r.get("epochs")
        ctx.adapter_lr = r.get("learning_rate")


# ── Step 4: Per-site adapter ────────────────────────────────────────────
class Step4Adapter(Step):
    index = 4
    title = "Per-site adapter"
    subtitle = "fine-tune last 2 layers"
    relevant_tools = ("finetune_adapter",)
    system_prompt = STEP_4_ADAPTER

    def required(self, ctx: OnboardingContext) -> list[str]:
        return ["site_id", "adapter_strategy", "user_confirmed_adapter", "ambient_source"]

    def run(self, ctx: OnboardingContext, agent: AgentSink, ui: UISink) -> StepResult:
        ui.step_header(self.index, 8, self.title, self.subtitle)
        self.install(agent)

        agent.user(
            f"Please call finetune_adapter(site_id='{ctx.site_id}', "
            f"ambient_source='{ctx.ambient_source}'). "
            f"Then report the final val_acc in 1 sentence."
        )
        if not _drive_turn(agent, ui, ctx, _extract_step4):
            return StepResult(StepStatus.NEEDS_RETRY)

        if ctx.adapter_val_acc is None:
            return StepResult(StepStatus.NEEDS_RETRY)
        return StepResult(StepStatus.OK)


def _extract_step4(ev: Any, ctx: OnboardingContext) -> None:
    if not isinstance(ev, ToolCallResult) or not ev.result.get("ok"):
        return
    r = ev.result
    if ev.name == "finetune_adapter":
        ctx.adapter_val_acc = r.get("final_val_acc")
        ctx.adapter_checkpoint = r.get("checkpoint")


# ── Step 5: Conformal calibration ───────────────────────────────────────
class Step5Conformal(Step):
    index = 5
    title = "Conformal calibration"
    subtitle = "false-alarm budget"
    relevant_tools = ("calibrate_conformal",)
    system_prompt = STEP_5_CONFORMAL

    def required(self, ctx: OnboardingContext) -> list[str]:
        return ["site_id", "adapter_val_acc", "ambient_source"]

    def run(self, ctx: OnboardingContext, agent: AgentSink, ui: UISink) -> StepResult:
        ui.step_header(self.index, 8, self.title, self.subtitle)
        self.install(agent)

        agent.user(
            f"Please call calibrate_conformal(site_id='{ctx.site_id}', "
            f"ambient_source='{ctx.ambient_source}', alpha=0.05). "
            f"Then report the threshold and coverage in 1-2 sentences."
        )
        if not _drive_turn(agent, ui, ctx, _extract_step5):
            return StepResult(StepStatus.NEEDS_RETRY)

        if ctx.conformal_threshold_p is None:
            return StepResult(StepStatus.NEEDS_RETRY)
        return StepResult(StepStatus.OK)


def _extract_step5(ev: Any, ctx: OnboardingContext) -> None:
    if not isinstance(ev, ToolCallResult) or not ev.result.get("ok"):
        return
    r = ev.result
    if ev.name == "calibrate_conformal":
        ctx.conformal_threshold_p = r.get("threshold_p")
        ctx.conformal_coverage = r.get("coverage")
        ctx.expected_fa_per_hour = r.get("expected_fa_per_hour")


# ── Step 6: Alert policy ────────────────────────────────────────────────
class Step6Policy(Step):
    index = 6
    title = "Alert policy"
    subtitle = "sensitivity + channels"
    relevant_tools = ("set_alert_policy",)
    system_prompt = STEP_6_POLICY

    def required(self, ctx: OnboardingContext) -> list[str]:
        return ["site_id"]

    def run(self, ctx: OnboardingContext, agent: AgentSink, ui: UISink) -> StepResult:
        ui.step_header(self.index, 8, self.title, self.subtitle)
        self.install(agent)

        sensitivity = ui.ask("Sensitivity?",
                             hint="high · medium · low",
                             default="medium").lower()
        if sensitivity not in {"high", "medium", "low"}:
            ui.warn(f"unknown sensitivity '{sensitivity}', defaulting to medium")
            sensitivity = "medium"
        ctx.sensitivity = sensitivity

        email = ui.ask("Alert email", hint="leave blank to skip")
        ctx.alert_email = email or None

        agent.user(
            f"The user chose sensitivity={sensitivity}, email={email or '(none)'}. "
            f"Please call set_alert_policy(site_id='{ctx.site_id}', "
            f"sensitivity='{sensitivity}', email='{email}'). Confirm in 1 sentence."
        )
        if not _drive_turn(agent, ui, ctx, _extract_step6):
            return StepResult(StepStatus.NEEDS_RETRY)

        if ctx.threshold_adjust is None:
            return StepResult(StepStatus.NEEDS_RETRY)
        return StepResult(StepStatus.OK)


def _extract_step6(ev: Any, ctx: OnboardingContext) -> None:
    if not isinstance(ev, ToolCallResult) or not ev.result.get("ok"):
        return
    r = ev.result
    if ev.name == "set_alert_policy":
        ctx.threshold_adjust = r.get("threshold_adjust")


# ── Step 7: Test detection ──────────────────────────────────────────────
class Step7Test(Step):
    index = 7
    title = "Test detection"
    subtitle = "run pipeline on a sample"
    relevant_tools = ("simulate_detection", "explain_decision")
    system_prompt = STEP_7_TEST

    def required(self, ctx: OnboardingContext) -> list[str]:
        return ["site_id", "adapter_val_acc", "conformal_threshold_p"]

    def run(self, ctx: OnboardingContext, agent: AgentSink, ui: UISink) -> StepResult:
        ui.step_header(self.index, 8, self.title, self.subtitle)
        self.install(agent)

        clip = ui.ask("Test clip path",
                      hint="path to a .wav to run through the pipeline",
                      default="data/sample/test_clip.wav")
        if not clip:
            return StepResult(StepStatus.NEEDS_RETRY)

        agent.user(
            f"Please call simulate_detection(site_id='{ctx.site_id}', "
            f"clip='{clip}'). Then call explain_decision with the returned "
            f"decision_id and modality='spectrogram+text' (this renders the "
            f"clip's spectrogram and runs your own multimodal vision over it). "
            f"Pass through the explanation field verbatim — do not paraphrase "
            f"or add invented details."
        )
        if not _drive_turn(agent, ui, ctx, _extract_step7):
            return StepResult(StepStatus.NEEDS_RETRY)

        if ctx.test_decision_id is None:
            return StepResult(StepStatus.NEEDS_RETRY)
        return StepResult(StepStatus.OK)


def _extract_step7(ev: Any, ctx: OnboardingContext) -> None:
    if not isinstance(ev, ToolCallResult) or not ev.result.get("ok"):
        return
    r = ev.result
    if ev.name == "simulate_detection":
        ctx.test_decision_id = r.get("decision_id")
        ctx.test_decision_tier = r.get("decision_tier")
        ctx.test_decision_severity = r.get("severity")
    elif ev.name == "explain_decision":
        ctx.test_explain_summary = r.get("summary")


# ── Step 8: Register ────────────────────────────────────────────────────
class Step8Register(Step):
    index = 8
    title = "Register site"
    subtitle = "save configuration"
    relevant_tools = ("register_site",)
    system_prompt = STEP_8_REGISTER

    def required(self, ctx: OnboardingContext) -> list[str]:
        # Earlier-step prerequisites are enforced by Steps 1..7 themselves.
        return ["site_id", "lat", "lon"]

    def run(self, ctx: OnboardingContext, agent: AgentSink, ui: UISink) -> StepResult:
        """Pure-Python step. Registration is fully deterministic at this point —
        we have all the data in ctx, just need to write the YAML. No LLM needed.

        This is also the canonical "fallback to Python" pattern: when a step
        is deterministic, bypass the model. Faster, more reliable, easier to
        audit. Gemma still narrates the result via a templated chat bubble."""
        ui.step_header(self.index, 8, self.title, self.subtitle)
        self.install(agent)   # for visual continuity (allowed_tools, prompt)

        from ..tools import dispatch
        config = ctx.to_yaml_config()
        ui.tool_call_static("register_site", {"site_id": ctx.site_id, "config": "<resolved>"})
        result = dispatch("register_site", {"site_id": ctx.site_id, "config": config})

        if not result.get("ok"):
            ui.fail(result.get("error", "register_site failed"))
            return StepResult(StepStatus.HARD_FAIL,
                              message=str(result.get("error", "register_site failed")))

        ctx.registered_yaml_path = result["path"]
        ui.ok(result["summary"])
        ui.gemma_say(
            f"Site `{ctx.site_id}` is now live. Configuration saved to "
            f"{result['path']}. You can start monitoring with "
            f"`os monitor {ctx.site_id}`."
        )
        return StepResult(StepStatus.OK)


# ── Registry ────────────────────────────────────────────────────────────
ALL_STEPS: list[Step] = [
    Step1Discovery(),
    Step2Ambient(),
    Step3Transfer(),
    Step4Adapter(),
    Step5Conformal(),
    Step6Policy(),
    Step7Test(),
    Step8Register(),
]
