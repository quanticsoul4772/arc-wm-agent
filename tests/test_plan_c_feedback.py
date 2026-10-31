"""WM_MECHANISMS=all-nocont-fb: the verdict after every cell that defines predict (not only a refused one), the
backtest's error text when it fails to run, and a wrong board shape named as such. In arms A-C 66-126 failed
captures per arm came from cells that were not refused, and the model was never told."""
import json
from dataclasses import replace

import pytest

from tests.fake_openai import FakeOpenAIServer, tool_call_response
from tests.test_agent import PREDICT_CELL, FakeGame, FakeRefused, FakeSandbox, _analyze
from tests.test_world_model_gate import _board, _run_backtest, _state
from wmagent.agent import CONFIG, ToolAgent
from wmagent.world_model_gate import BACKTEST_SCOPE, WorldModelGate, mechanisms_from_env

DOWN_BOARDS = [_board(r) for r in range(6)]
ONE_LINE_PREDICT = '''
def predict(ascii, action):
    return ascii.replace("\\n", "")
'''


def _gate(tmp_path, mechanisms="all-nocont-fb"):
    gate = WorldModelGate(probe_actions=8, max_refusals=8, mechanisms=mechanisms)
    tmp_path.mkdir(parents=True, exist_ok=True)
    gate.reset(tmp_path / "g_p0_tool_runtime_state.json")
    gate.observe_level(1)
    return gate


def test_driver_reports_a_wrong_board_shape_and_nothing_for_the_right_one():
    state = _state(DOWN_BOARDS, ["DOWN"] * 6)
    wrong = _run_backtest(ONE_LINE_PREDICT, state)
    assert wrong and all(row[7] == [1, 64, 8, 8] for row in wrong)
    right = _run_backtest("def predict(ascii, action):\n    return ascii\n", state)
    assert right and all(row[7] is None for row in right)


def test_fb_names_the_wrong_shape_in_the_verdict(tmp_path):
    rows = _run_backtest(ONE_LINE_PREDICT, _state(DOWN_BOARDS, ["DOWN"] * 6))
    gate = _gate(tmp_path)
    gate.after_cell(ONE_LINE_PREDICT, False, lambda src: rows)
    note = gate.verdict_note()
    assert "returned 1 line(s) up to 64 characters wide; the board is 8 lines of 8" in note
    assert "returned 1 line(s)" in gate.evidence_line() and len(gate.evidence_line()) <= 250
    other = _gate(tmp_path / "c2", mechanisms="all-nocont")
    other.after_cell(ONE_LINE_PREDICT, False, lambda src: rows)
    assert "returned" not in other.verdict_note() and "First miss" not in other.verdict_note()
    assert "was off by" in other.evidence_line()  # the tested all-nocont text is unchanged


def test_fb_gives_the_backtest_error_and_the_scope_rule(tmp_path):
    def failing(src):
        gate.note_backtest_error('Traceback (most recent call last):\n  File "<python_tool>", line 9, in <module>\n'
                                 "NameError: name 'predicted' is not defined")

    gate = _gate(tmp_path)
    gate.after_cell("def predict(ascii, action):\n    return ascii\n", False, failing)
    note = gate.verdict_note()
    assert "NameError: name 'predicted' is not defined" in note and BACKTEST_SCOPE in note
    assert "NameError" in gate.evidence_line()
    logged = [json.loads(x) for x in gate.log_path.read_text().splitlines()]
    assert [e for e in logged if e["event"] == "captured"][-1]["error"].startswith("NameError")
    gate = _gate(tmp_path / "c2", mechanisms="all-nocont")
    gate.after_cell("def predict(ascii, action):\n    return ascii\n", False, failing)
    assert gate.verdict_note() == "World-model gate: your predict was captured but its backtest failed to run."


def _first_cell_tool_result(tmp_path, mechanisms):
    game = FakeGame(tmp_path)
    gate = WorldModelGate(probe_actions=8, max_refusals=8, mechanisms=mechanisms)
    script = [tool_call_response(PREDICT_CELL), tool_call_response('action("UP")')]
    with FakeOpenAIServer(script) as server:
        config = replace(CONFIG, base_url=server.base_url, mechanisms=mechanisms)
        agent = ToolAgent(config=config, gate=gate, sandbox=(FakeSandbox(), FakeRefused))
        _analyze(agent, game, 1)
        tools = [json.loads(m["content"]) for m in server.requests[1]["messages"] if m["role"] == "tool"]
    return tools[-1]


def test_fb_tells_the_model_the_verdict_after_a_cell_that_was_not_refused(tmp_path):
    tool = _first_cell_tool_result(tmp_path / "fb", "all-nocont-fb")
    assert "error" not in tool or not tool["error"]
    assert tool["gate"].startswith("World-model gate: your predict was captured")
    for arm in ("base", "m1a", "all", "all-nocont"):
        assert "gate" not in _first_cell_tool_result(tmp_path / arm, arm)


def test_fb_is_a_full_arm_without_continuation_and_a_valid_build_value(tmp_path, monkeypatch):
    from wmagent.kaggle.build import agent_env, main, parse_args

    gate = WorldModelGate(probe_actions=8, max_refusals=8, mechanisms="all-nocont-fb")
    assert gate.full and gate.capture_feedback
    assert not WorldModelGate(probe_actions=8, max_refusals=8, mechanisms="all-nocont").capture_feedback
    monkeypatch.setenv("WM_MECHANISMS", "all-nocont-fb")
    assert mechanisms_from_env() == "all-nocont-fb"
    assert agent_env(parse_args(["--out", "o", "--label", "t", "--wm-mechanisms", "all-nocont-fb"]))[
        "WM_MECHANISMS"] == "all-nocont-fb"
    main(["--out", str(tmp_path), "--label", "t", "--wm-mechanisms", "all-nocont-fb", "--dry-run"])
    env = json.loads((tmp_path / "kaggle" / "source-dataset" / "agent_env.json").read_text())
    assert env["WM_MECHANISMS"] == "all-nocont-fb"
    with pytest.raises(SystemExit):
        parse_args(["--out", "o", "--label", "t", "--wm-mechanisms", "fb"])


def test_fb_continuation_stays_off(tmp_path):
    from tests.test_plan_c_step2 import _user_texts, _yield_then_play

    _, second, requests, _ = _yield_then_play(tmp_path, "all-nocont-fb")
    assert second["step_executed"] and "Your turn is still open" not in _user_texts(requests[1])[-1]
