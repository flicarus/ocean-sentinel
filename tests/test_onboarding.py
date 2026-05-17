"""Onboarding state-machine tests with MockAgent + MockUI.

Tests cover:
- Per-step happy paths (1-8) — each step extracts the right context fields
- Per-step error paths — invalid input → NEEDS_RETRY; user 'no' → USER_ABORT
- Flow orchestration — retries, hard fail after 3 attempts, audit log
- Context serialization — checkpoint save/load round-trip, resume detection

The tests do NOT call Ollama. MockAgent yields scripted events; MockUI
records calls and pops scripted replies. This keeps tests fast and
deterministic, and lets CI run without an Ollama server.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from ocean_sentinel.gemma.agent import (
    AgentError,
    GemmaText,
    ToolCallResult,
    ToolCallStart,
)
from ocean_sentinel.gemma.onboarding import (
    OnboardingContext,
    OnboardingFlow,
    StepResult,
    StepStatus,
)
from ocean_sentinel.gemma.onboarding.steps import (
    ALL_STEPS,
    Step1Discovery,
    Step2Ambient,
    Step3Transfer,
    Step4Adapter,
    Step5Conformal,
    Step6Policy,
    Step7Test,
    Step8Register,
)


# ── MockAgent ───────────────────────────────────────────────────────────
class MockAgent:
    """Yields scripted events instead of calling Ollama. Tracks tool args
    Gemma 'requested' so tests can assert on prompt routing too."""

    def __init__(self) -> None:
        self.messages: list[dict[str, Any]] = []
        self.tools: list[dict[str, Any]] = []
        self._turn_queue: list[list[Any]] = []

    def system(self, prompt: str) -> None:
        self.messages.append({"role": "system", "content": prompt})

    def user(self, content: str) -> None:
        self.messages.append({"role": "user", "content": content})

    def script(self, *events: Any) -> "MockAgent":
        """Queue one turn worth of events."""
        self._turn_queue.append(list(events))
        return self

    def turn(self):
        if not self._turn_queue:
            return iter([])
        events = self._turn_queue.pop(0)
        return iter(events)


# ── MockUI ──────────────────────────────────────────────────────────────
class MockUI:
    def __init__(self, replies: list[str] | None = None,
                 confirms: list[bool] | None = None) -> None:
        self.replies = list(replies or [])
        self.confirms = list(confirms or [])
        self.calls: list[tuple] = []

    # noinspection PyUnusedLocal
    def step_header(self, n, total, title, subtitle=""):
        self.calls.append(("step_header", n, title))

    def gemma_say(self, text):
        self.calls.append(("gemma_say", text))

    def ask(self, question, hint="", default=None):
        self.calls.append(("ask", question))
        if not self.replies:
            return default or ""
        return self.replies.pop(0)

    def confirm(self, question, default_yes=True):
        self.calls.append(("confirm", question))
        if not self.confirms:
            return default_yes
        return self.confirms.pop(0)

    def tool_call_static(self, name, args):
        self.calls.append(("tool_call_static", name))

    def ok(self, text):
        self.calls.append(("ok", text))

    def fail(self, text):
        self.calls.append(("fail", text))

    def warn(self, text):
        self.calls.append(("warn", text))

    def error(self, text):
        self.calls.append(("error", text))


# ── Helpers ─────────────────────────────────────────────────────────────
def _ok_result(name: str, **fields) -> ToolCallResult:
    return ToolCallResult(name=name, result={"ok": True, **fields})


# ── Step 1: Discovery ───────────────────────────────────────────────────
class TestStep1Discovery:
    def test_happy_path(self):
        ctx = OnboardingContext()
        ui = MockUI(replies=["36.7128, -121.9023", "monterey-test"])
        agent = MockAgent().script(
            ToolCallStart(name="validate_site_coords",
                          arguments={"lat": 36.7128, "lon": -121.9023}),
            _ok_result("validate_site_coords",
                       summary="ok", depth_m=520, nearest_mpa="MBNMS"),
            ToolCallStart(name="fetch_ais_baseline",
                          arguments={"lat": 36.7128, "lon": -121.9023,
                                     "radius_km": 10, "days": 30}),
            _ok_result("fetch_ais_baseline",
                       summary="ok", traffic_class="high",
                       avg_vessels_per_day=47, shipping_lane_km=11.0),
            GemmaText(content="Step 1 complete."),
        )

        result = Step1Discovery().run(ctx, agent, ui)

        assert result.status == StepStatus.OK
        assert ctx.lat == 36.7128
        assert ctx.lon == -121.9023
        assert ctx.site_id == "monterey-test"
        assert ctx.depth_m == 520
        assert ctx.nearest_mpa == "MBNMS"
        assert ctx.ais_traffic_class == "high"
        assert ctx.ais_avg_vessels_per_day == 47

    def test_invalid_coords_rejected(self):
        ctx = OnboardingContext()
        ui = MockUI(replies=["not a coordinate", ""])
        agent = MockAgent()

        result = Step1Discovery().run(ctx, agent, ui)
        assert result.status == StepStatus.NEEDS_RETRY
        assert ctx.lat is None

    def test_out_of_range_lat(self):
        ctx = OnboardingContext()
        ui = MockUI(replies=["91.0, 0.0", ""])
        agent = MockAgent()

        result = Step1Discovery().run(ctx, agent, ui)
        assert result.status == StepStatus.NEEDS_RETRY

    def test_invalid_site_id_rejected(self):
        ctx = OnboardingContext()
        ui = MockUI(replies=["36.7, -121.9", "Bad Site Name"])
        agent = MockAgent()

        result = Step1Discovery().run(ctx, agent, ui)
        assert result.status == StepStatus.NEEDS_RETRY
        assert ctx.site_id == ""

    def test_tool_returned_no_data(self):
        """If validate_site_coords runs but doesn't populate depth_m, retry."""
        ctx = OnboardingContext()
        ui = MockUI(replies=["36.7, -121.9", "good-site"])
        # tool result is "ok" but missing fields — extractor sets None values
        agent = MockAgent().script(
            ToolCallStart(name="validate_site_coords", arguments={}),
            _ok_result("validate_site_coords"),  # no depth_m
            GemmaText(content="hmm"),
        )
        result = Step1Discovery().run(ctx, agent, ui)
        assert result.status == StepStatus.NEEDS_RETRY


# ── Step 2: Ambient ─────────────────────────────────────────────────────
class TestStep2Ambient:
    def test_happy_path(self):
        ctx = OnboardingContext(site_id="monterey-test")
        ui = MockUI(replies=["monterey_5min.wav"])
        sig = [-72.0] * 64
        agent = MockAgent().script(
            ToolCallStart(name="record_ambient",
                          arguments={"source": "monterey_5min.wav"}),
            _ok_result("record_ambient", summary="loaded"),
            ToolCallStart(name="compute_spectral_signature", arguments={}),
            _ok_result("compute_spectral_signature",
                       signature=sig,
                       ambient_class="deep-water",
                       dominant_band_hz="80-200"),
            GemmaText(content="Step 2 complete."),
        )

        result = Step2Ambient().run(ctx, agent, ui)
        assert result.status == StepStatus.OK
        assert ctx.spectral_signature == sig
        assert ctx.ambient_class == "deep-water"

    def test_blank_source_retries(self):
        ctx = OnboardingContext(site_id="monterey-test")
        ui = MockUI(replies=[""])
        agent = MockAgent()
        result = Step2Ambient().run(ctx, agent, ui)
        assert result.status == StepStatus.NEEDS_RETRY


# ── Step 3: Transfer ────────────────────────────────────────────────────
class TestStep3Transfer:
    def _make_ctx(self):
        return OnboardingContext(
            site_id="monterey-test",
            spectral_signature=[-72.0] * 64,
        )

    def test_happy_path(self):
        ctx = self._make_ctx()
        ui = MockUI(confirms=[True])
        agent = MockAgent().script(
            ToolCallStart(name="compare_to_known_sites", arguments={}),
            _ok_result("compare_to_known_sites",
                       ranked=[{"id": "mbari-mars", "label": "MBARI MARS",
                                "cosine_sim": 0.84}],
                       recommendation="finetune"),
            ToolCallStart(name="select_adapter_strategy", arguments={}),
            _ok_result("select_adapter_strategy",
                       strategy="finetune_last2",
                       epochs=10, learning_rate=3e-4),
            GemmaText(content="Recommend finetune."),
        )

        result = Step3Transfer().run(ctx, agent, ui)
        assert result.status == StepStatus.OK
        assert ctx.adapter_strategy == "finetune_last2"
        assert ctx.adapter_epochs == 10
        assert ctx.user_confirmed_adapter is True

    def test_user_declines(self):
        ctx = self._make_ctx()
        ui = MockUI(confirms=[False])
        agent = MockAgent().script(
            ToolCallStart(name="compare_to_known_sites", arguments={}),
            _ok_result("compare_to_known_sites",
                       ranked=[{"id": "x", "label": "x", "cosine_sim": 0.5}],
                       recommendation="finetune"),
            ToolCallStart(name="select_adapter_strategy", arguments={}),
            _ok_result("select_adapter_strategy",
                       strategy="finetune_last2", epochs=10, learning_rate=3e-4),
            GemmaText(content="ask user"),
        )

        result = Step3Transfer().run(ctx, agent, ui)
        assert result.status == StepStatus.USER_ABORT
        assert ctx.user_confirmed_adapter is False


# ── Step 4: Adapter ─────────────────────────────────────────────────────
class TestStep4Adapter:
    def test_happy_path(self):
        ctx = OnboardingContext(
            site_id="monterey-test",
            adapter_strategy="finetune_last2",
            adapter_epochs=10,
            adapter_lr=3e-4,
            user_confirmed_adapter=True,
        )
        ui = MockUI()
        agent = MockAgent().script(
            ToolCallStart(name="finetune_adapter", arguments={}),
            _ok_result("finetune_adapter",
                       final_val_acc=0.892,
                       checkpoint="data/sites/monterey-test/adapter.pt"),
            GemmaText(content="Step 4 done."),
        )

        result = Step4Adapter().run(ctx, agent, ui)
        assert result.status == StepStatus.OK
        assert ctx.adapter_val_acc == 0.892


# ── Step 5: Conformal ───────────────────────────────────────────────────
class TestStep5Conformal:
    def test_happy_path(self):
        ctx = OnboardingContext(site_id="monterey-test", adapter_val_acc=0.89)
        ui = MockUI()
        agent = MockAgent().script(
            ToolCallStart(name="calibrate_conformal", arguments={}),
            _ok_result("calibrate_conformal",
                       threshold_p=0.71, coverage=0.954,
                       expected_fa_per_hour=0.001),
            GemmaText(content="Step 5 done."),
        )

        result = Step5Conformal().run(ctx, agent, ui)
        assert result.status == StepStatus.OK
        assert ctx.conformal_threshold_p == 0.71


# ── Step 6: Policy ──────────────────────────────────────────────────────
class TestStep6Policy:
    def test_happy_path(self):
        ctx = OnboardingContext(site_id="monterey-test")
        ui = MockUI(replies=["medium", "alerts@noaa.gov"])
        agent = MockAgent().script(
            ToolCallStart(name="set_alert_policy", arguments={}),
            _ok_result("set_alert_policy", threshold_adjust=0.0),
            GemmaText(content="Step 6 done."),
        )
        result = Step6Policy().run(ctx, agent, ui)
        assert result.status == StepStatus.OK
        assert ctx.sensitivity == "medium"
        assert ctx.alert_email == "alerts@noaa.gov"

    def test_unknown_sensitivity_defaults(self):
        ctx = OnboardingContext(site_id="monterey-test")
        ui = MockUI(replies=["bogus", ""])
        agent = MockAgent().script(
            ToolCallStart(name="set_alert_policy", arguments={}),
            _ok_result("set_alert_policy", threshold_adjust=0.0),
            GemmaText(content="ok"),
        )
        result = Step6Policy().run(ctx, agent, ui)
        assert result.status == StepStatus.OK
        assert ctx.sensitivity == "medium"

    def test_blank_email_stored_as_none(self):
        ctx = OnboardingContext(site_id="monterey-test")
        ui = MockUI(replies=["high", ""])
        agent = MockAgent().script(
            ToolCallStart(name="set_alert_policy", arguments={}),
            _ok_result("set_alert_policy", threshold_adjust=-0.05),
            GemmaText(content="ok"),
        )
        Step6Policy().run(ctx, agent, ui)
        assert ctx.alert_email is None


# ── Step 7: Test detection ──────────────────────────────────────────────
class TestStep7Test:
    def test_happy_path(self):
        ctx = OnboardingContext(
            site_id="monterey-test",
            adapter_val_acc=0.89,
            conformal_threshold_p=0.71,
        )
        ui = MockUI(replies=["test_clip.wav"])
        agent = MockAgent().script(
            ToolCallStart(name="simulate_detection", arguments={}),
            _ok_result("simulate_detection",
                       decision_id="DRY-RUN-1234",
                       decision_tier="DARK_VESSEL",
                       severity="HIGH"),
            ToolCallStart(name="explain_decision", arguments={}),
            _ok_result("explain_decision",
                       summary="trawl-class signature, 18 dB above ambient"),
            GemmaText(content="The model heard..."),
        )

        result = Step7Test().run(ctx, agent, ui)
        assert result.status == StepStatus.OK
        assert ctx.test_decision_id == "DRY-RUN-1234"
        assert ctx.test_decision_tier == "DARK_VESSEL"


# ── Step 8: Register (pure Python, no agent) ────────────────────────────
class TestStep8Register:
    def test_writes_yaml(self, tmp_path, monkeypatch):
        from ocean_sentinel.gemma import tools as tools_mod
        monkeypatch.setattr(tools_mod, "SITES_DIR", tmp_path)

        ctx = OnboardingContext(
            site_id="monterey-test",
            lat=36.7128, lon=-121.9023,
            depth_m=520,
            ambient_class="deep-water",
        )
        ui = MockUI()
        agent = MockAgent()  # never called

        result = Step8Register().run(ctx, agent, ui)
        assert result.status == StepStatus.OK
        assert ctx.registered_yaml_path is not None
        yaml_path = Path(ctx.registered_yaml_path)
        assert yaml_path.exists()
        # We never called agent — this step is pure Python.
        assert agent.messages == [
            # only the system prompt installed by Step.install
            m for m in agent.messages if m.get("role") == "system"
        ]


# ── Context serialization ───────────────────────────────────────────────
class TestContext:
    def test_roundtrip(self, tmp_path, monkeypatch):
        from ocean_sentinel.gemma.onboarding import context as ctx_mod
        monkeypatch.setattr(ctx_mod, "SESSIONS_DIR", tmp_path)

        ctx = OnboardingContext(
            site_id="abc",
            lat=1.5, lon=2.5,
            depth_m=400,
            spectral_signature=[1.0, 2.0, 3.0],
            adapter_val_acc=0.91,
            sensitivity="high",
            last_completed_step=5,
        )
        ctx.checkpoint()

        loaded = OnboardingContext.load("abc")
        assert loaded is not None
        assert loaded.site_id == "abc"
        assert loaded.lat == 1.5
        assert loaded.depth_m == 400
        assert loaded.adapter_val_acc == 0.91
        assert loaded.last_completed_step == 5
        assert loaded.spectral_signature == [1.0, 2.0, 3.0]

    def test_load_missing_returns_none(self, tmp_path, monkeypatch):
        from ocean_sentinel.gemma.onboarding import context as ctx_mod
        monkeypatch.setattr(ctx_mod, "SESSIONS_DIR", tmp_path)
        assert OnboardingContext.load("does-not-exist") is None

    def test_to_yaml_drops_none_and_empty(self):
        ctx = OnboardingContext(site_id="x", lat=1.0, lon=2.0,
                                depth_m=None, spectral_signature=[])
        out = ctx.to_yaml_config()
        assert "site_id" in out
        assert "lat" in out
        assert "depth_m" not in out
        assert "spectral_signature" not in out

    def test_list_resumable_filters_complete(self, tmp_path, monkeypatch):
        from ocean_sentinel.gemma.onboarding import context as ctx_mod
        monkeypatch.setattr(ctx_mod, "SESSIONS_DIR", tmp_path)

        OnboardingContext(site_id="incomplete", last_completed_step=4).checkpoint()
        OnboardingContext(site_id="finished",   last_completed_step=8).checkpoint()

        resumable = OnboardingContext.list_resumable()
        site_ids = {sid for sid, _ in resumable}
        assert "incomplete" in site_ids
        assert "finished" not in site_ids


# ── Flow orchestration ──────────────────────────────────────────────────
class _AlwaysFailStep:
    """Step that always returns NEEDS_RETRY — for testing retry logic."""
    index = 1
    title = "Always Fail"
    subtitle = ""
    relevant_tools = ()
    system_prompt = ""
    attempts = 0

    def required(self, ctx): return []
    def validate_preconditions(self, ctx): return []
    def install(self, agent): pass
    def filtered_tools(self, all_tools): return []

    def run(self, ctx, agent, ui):
        self.attempts += 1
        return StepResult(StepStatus.NEEDS_RETRY, message="planned failure")


class _SucceedAtN:
    """Succeeds on the Nth attempt."""
    index = 1
    title = "Succeed at N"
    subtitle = ""
    relevant_tools = ()
    system_prompt = ""

    def __init__(self, succeed_at: int):
        self.succeed_at = succeed_at
        self.attempts = 0

    def required(self, ctx): return []
    def validate_preconditions(self, ctx): return []
    def install(self, agent): pass
    def filtered_tools(self, all_tools): return []

    def run(self, ctx, agent, ui):
        self.attempts += 1
        if self.attempts >= self.succeed_at:
            ctx.last_completed_step = 1
            return StepResult(StepStatus.OK)
        return StepResult(StepStatus.NEEDS_RETRY, message=f"attempt {self.attempts}")


class TestFlow:
    def test_flow_runs_step_3_times_then_hard_fails(self, tmp_path, monkeypatch):
        from ocean_sentinel.gemma.onboarding import context as ctx_mod
        monkeypatch.setattr(ctx_mod, "SESSIONS_DIR", tmp_path)

        step = _AlwaysFailStep()
        flow = OnboardingFlow(MockAgent(), MockUI(), steps=[step])
        flow.ctx.site_id = "test-site"
        flow.run()
        assert step.attempts == 3   # MAX_RETRIES_PER_STEP

    def test_flow_succeeds_after_two_retries(self, tmp_path, monkeypatch):
        from ocean_sentinel.gemma.onboarding import context as ctx_mod
        monkeypatch.setattr(ctx_mod, "SESSIONS_DIR", tmp_path)

        step = _SucceedAtN(succeed_at=3)
        flow = OnboardingFlow(MockAgent(), MockUI(), steps=[step])
        flow.ctx.site_id = "test-site"
        ctx = flow.run()
        assert step.attempts == 3
        assert ctx.last_completed_step == 1

    def test_flow_skips_already_completed_steps_on_resume(self, tmp_path, monkeypatch):
        from ocean_sentinel.gemma.onboarding import context as ctx_mod
        monkeypatch.setattr(ctx_mod, "SESSIONS_DIR", tmp_path)

        step = _SucceedAtN(succeed_at=1)
        ctx = OnboardingContext(site_id="resumed", last_completed_step=1)
        flow = OnboardingFlow(MockAgent(), MockUI(), steps=[step], ctx=ctx)
        flow.run()
        # Step's index is 1 and last_completed is already 1, so it should not run.
        assert step.attempts == 0

    def test_flow_writes_audit_log(self, tmp_path, monkeypatch):
        from ocean_sentinel.gemma.onboarding import context as ctx_mod
        from ocean_sentinel.gemma.onboarding import flow as flow_mod
        monkeypatch.setattr(ctx_mod, "SESSIONS_DIR", tmp_path)
        monkeypatch.setattr(flow_mod, "SESSIONS_DIR", tmp_path)

        step = _SucceedAtN(succeed_at=1)
        ctx = OnboardingContext(site_id="audit-test")
        flow = OnboardingFlow(MockAgent(), MockUI(), steps=[step], ctx=ctx)
        flow.run()

        log = tmp_path / "audit-test.events.jsonl"
        assert log.exists()
        events = [json.loads(line) for line in log.read_text().splitlines() if line]
        names = [e["event"] for e in events]
        assert "flow.start" in names
        assert "step.attempt" in names
        assert "step.result" in names
        assert "flow.complete" in names


# ── Smoke: every step has its prompt + tool subset wired ────────────────
class TestRegistry:
    def test_all_8_steps_registered_with_prompts(self):
        assert len(ALL_STEPS) == 8
        indices = [s.index for s in ALL_STEPS]
        assert indices == [1, 2, 3, 4, 5, 6, 7, 8]

    def test_each_step_has_nonempty_prompt(self):
        for step in ALL_STEPS:
            assert step.system_prompt.strip(), f"step {step.index} has empty prompt"
            assert step.title, f"step {step.index} has empty title"

    def test_each_step_restricts_tools(self):
        from ocean_sentinel.gemma.tools import all_schemas
        all_names = {s["function"]["name"] for s in all_schemas()}
        for step in ALL_STEPS:
            for tool_name in step.relevant_tools:
                assert tool_name in all_names, (
                    f"step {step.index} references unknown tool {tool_name}"
                )

    def test_step_install_filters_tools(self):
        from ocean_sentinel.gemma.tools import all_schemas
        agent = MockAgent()
        Step1Discovery().install(agent)
        names = {t["function"]["name"] for t in agent.tools}
        # Step 1 should only have its 3 tools, not all 14
        assert names == {"validate_site_coords", "fetch_ais_baseline",
                         "fetch_hydrophone_metadata"}
        assert len(names) < len(all_schemas())
