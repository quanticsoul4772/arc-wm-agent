"""Plan C step 1: capture-on-deny (M1a), the stated acceptance rule, WM_MECHANISMS and the boot manifest.

The "base" arm must behave exactly as before (the existing tests cover that); these tests cover the
"m1a" / "all" arms and the plumbing.
"""
import json
from dataclasses import replace

import pytest

from tests.fake_openai import FakeOpenAIServer, error_response, tool_call_response
from tests.test_agent import PREDICT_CELL, FakeGame, FakeRefused, FakeSandbox, _analyze
from wmagent import prompts
from wmagent.agent import CONFIG, ToolAgent
from wmagent.world_model_gate import CHANGE_FLOOR, MIN_INFORMATIVE, WorldModelGate, mechanisms_from_env

REFUSED_PREDICT_CELL = PREDICT_CELL + '\naction("UP")\n'
CAUGHT_THEN_FAILS = PREDICT_CELL + '\ntry:\n    action("UP")\nexcept RuntimeError:\n    pass\n1 / 0\n'


def _agent(server, gate, mechanisms):
    config = replace(CONFIG, base_url=server.base_url, mechanisms=mechanisms)
    return ToolAgent(config=config, gate=gate, sandbox=(FakeSandbox(), FakeRefused))


def _run(tmp_path, mechanisms, second_cell):
    """Spend the 1-action probe budget, then send `second_cell`, which calls action() and is refused."""
    game = FakeGame(tmp_path)
    gate = WorldModelGate(probe_actions=1, max_refusals=8, mechanisms=mechanisms)
    script = [tool_call_response('action("UP")'), tool_call_response(second_cell), error_response(503, "stop")]
    with FakeOpenAIServer(script) as server:
        agent = _agent(server, gate, mechanisms)
        assert _analyze(agent, game, 1)["step_executed"]
        _analyze(agent, game, 2)
        tools = [json.loads(m["content"]) for m in server.requests[2]["messages"] if m["role"] == "tool"]
    events = [json.loads(line) for line in gate.log_path.read_text().splitlines()]
    return events, tools[-1]


def test_m1a_captures_predict_from_a_refused_cell_and_reports_the_verdict(tmp_path):
    events, tool = _run(tmp_path, "m1a", REFUSED_PREDICT_CELL)
    names = [e["event"] for e in events]
    assert names.index("refused") < names.index("captured")
    assert "RuntimeError" in tool["error"] and "World-model gate" in tool["error"]
    # FakeGame moves change 2 cells, below the change floor: captured, judged, not accepted
    assert "captured but not accepted" in tool["gate"] and f"more than {CHANGE_FLOOR} cells" in tool["gate"]


def test_base_arm_still_drops_predict_from_a_refused_cell(tmp_path):
    events, tool = _run(tmp_path, "base", REFUSED_PREDICT_CELL)
    assert "refused" in [e["event"] for e in events]
    assert "captured" not in [e["event"] for e in events]
    assert "gate" not in tool


def test_m1a_caught_refusal_then_another_error_is_still_an_errored_cell(tmp_path):
    events, tool = _run(tmp_path, "m1a", CAUGHT_THEN_FAILS)
    assert "refused" in [e["event"] for e in events]
    assert "captured" not in [e["event"] for e in events]
    assert "ZeroDivisionError" in tool["error"] and "gate" not in tool


def test_m1a_reports_acceptance_when_the_refused_cells_predict_passes(tmp_path, monkeypatch):
    def passing(self, rows):
        return {"passed": True, "accuracy": 1.0, "informative": 3, "errors": 0, "n_transitions": 2}

    monkeypatch.setattr(WorldModelGate, "_judge", passing)
    events, tool = _run(tmp_path, "m1a", REFUSED_PREDICT_CELL)
    assert "accepted" in [e["event"] for e in events]
    assert "accepted on this level; action() is unblocked" in tool["gate"]


def test_verdict_note_counts_matched_moves():
    gate = WorldModelGate(probe_actions=8, max_refusals=8, mechanisms="m1a")
    assert gate.verdict_note() is None
    gate.last_verdict = {"passed": False, "accuracy": 0.5, "informative": 4, "errors": 0, "n_transitions": 9}
    assert "matched 2 of 4 informative moves" in gate.verdict_note()
    gate.last_verdict = {"passed": False, "reason": "backtest_failed", "n_transitions": 0}
    assert "backtest failed" in gate.verdict_note()


def test_m1a_prompt_states_the_rule_the_gate_enforces():
    base = prompts.system_prompt("forced", 8)
    m1a = prompts.system_prompt("forced", 8, "m1a")
    assert base == prompts.system_prompt("forced", 8, "base")
    assert f"at least {MIN_INFORMATIVE} such moves" in m1a and f"more than \n{CHANGE_FLOOR}" not in m1a
    assert f"changed more than {CHANGE_FLOOR} cells" in m1a
    assert "even if the cell's `action()` call was refused" in m1a
    assert "runs without an error" in base and "runs without an error" not in m1a
    assert prompts.system_prompt("forced", 8, "all") == m1a
    assert prompts.system_prompt("optional", 8, "m1a") == prompts.system_prompt("optional", 8)


def test_manifest_records_the_arm_and_the_enforced_values(tmp_path):
    gate = WorldModelGate(probe_actions=8, max_refusals=8, mechanisms="all")
    gate.reset(tmp_path / "g_p0_tool_runtime_state.json")
    start = json.loads(gate.log_path.read_text().splitlines()[0])
    assert start["event"] == "session_start" and start["mechanisms"] == "all"
    assert start["change_floor"] == CHANGE_FLOOR and start["min_informative"] == MIN_INFORMATIVE


def test_wm_mechanisms_env_is_validated_and_read_by_gate_and_config(monkeypatch):
    monkeypatch.delenv("WM_MECHANISMS", raising=False)
    assert mechanisms_from_env() == "base"
    monkeypatch.setenv("WM_MECHANISMS", "M1A")
    assert mechanisms_from_env() == "m1a"
    monkeypatch.setenv("PROBE_ENFORCE", "1")
    assert WorldModelGate.from_env().mechanisms == "m1a"
    monkeypatch.setenv("WM_MECHANISMS", "everything")
    with pytest.raises(ValueError, match="WM_MECHANISMS"):
        mechanisms_from_env()


def test_kaggle_env_carries_wm_mechanisms():
    from wmagent.kaggle.build import agent_env, parse_args

    assert agent_env(parse_args(["--out", "o", "--label", "t"]))["WM_MECHANISMS"] == "base"
    assert agent_env(parse_args(["--out", "o", "--label", "t", "--wm-mechanisms", "all"]))["WM_MECHANISMS"] == "all"
    with pytest.raises(SystemExit):
        parse_args(["--out", "o", "--label", "t", "--wm-mechanisms", "x"])
