"""WM_MECHANISMS=all-nocont-fbx: all-nocont-fb plus four fixes found in C200's logs.

- A rejected action (e.g. ACTION7, listed but unplayable) costs no probe (it cost up to 4 of 8 probes).
- A release (valve or refusal cap) is announced in the next prompts (the system prompt says it cannot happen).
- Traceback line numbers count from the model's own cell, not from the stored definitions run above it.
- Capture keeps only constants whose names are bound (all 29 backtest failures were NameErrors from constants
  built from lines the capture drops)."""
import json
from dataclasses import replace

from tests.fake_openai import FakeOpenAIServer, error_response, tool_call_response
from tests.test_agent import FakeGame, FakeRefused, FakeSandbox, _analyze
from wmagent.agent import CONFIG, ToolAgent, _cell_line_numbers
from wmagent.world_model_gate import WorldModelGate, extract_definitions, stored_names


def _agent(server, gate, mechanisms):
    config = replace(CONFIG, base_url=server.base_url, mechanisms=mechanisms)
    return ToolAgent(config=config, gate=gate, sandbox=(FakeSandbox(), FakeRefused))


def test_a_rejected_action_costs_a_probe_only_outside_fbx(tmp_path):
    for arm, expected in (("all-nocont-fb", 1), ("all-nocont-fbx", 0)):
        game = FakeGame(tmp_path / arm)
        gate = WorldModelGate(probe_actions=8, max_refusals=8, mechanisms=arm)
        with FakeOpenAIServer([tool_call_response('action("ACTION7")'), error_response(503, "stop")]) as server:
            _analyze(_agent(server, gate, arm), game, 1)
        assert gate.level_actions == expected, arm


def test_a_release_is_announced_in_fbx_only(tmp_path):
    for arm in ("all-nocont-fb", "all-nocont-fbx"):
        gate = WorldModelGate(probe_actions=1, max_refusals=8, mechanisms=arm)
        (tmp_path / arm).mkdir()
        gate.reset(tmp_path / arm / "g_p0_tool_runtime_state.json")
        gate.observe_level(1)
        gate.record_actions(1)
        gate.check_action(1)
        assert gate.released_note(1) is None  # blocked, not released
        for _ in range(8):
            gate.note_model_call()
        assert gate.released
        note = gate.released_note(1)
        if arm == "all-nocont-fbx":
            assert note.startswith("World-model gate: action() is unblocked on this level")
            assert gate.released_note(2) is None  # the board moved on
        else:
            assert note is None


def test_a_released_note_reaches_the_prompt(tmp_path):
    game = FakeGame(tmp_path)
    gate = WorldModelGate(probe_actions=1, max_refusals=8, mechanisms="all-nocont-fbx")
    script = [tool_call_response('action("UP")'), tool_call_response('action("UP")')] + [
        tool_call_response('print("x")')] * 8 + [tool_call_response('action("UP")')]
    with FakeOpenAIServer(script + [error_response(503, "stop")]) as server:
        agent = _agent(server, gate, "all-nocont-fbx")
        _analyze(agent, game, 1)
        _analyze(agent, game, 2)
        _analyze(agent, game, 3)
        users = []
        for request in server.requests:
            user = [m for m in request["messages"] if m["role"] == "user"][-1]
            users.append(user["content"][0]["text"] if isinstance(user["content"], list) else user["content"])
    assert gate.released
    assert any("action() is unblocked on this level" in u for u in users)


def test_traceback_lines_count_from_the_models_cell():
    error = ('Traceback (most recent call last):\n  File "<python_tool>", line 14, in <module>\n'
             '  File "<python_tool>", line 3, in predict\nNameError: name \'x\' is not defined')
    fixed = _cell_line_numbers(error, 10)
    assert 'line 4, in <module>' in fixed and 'line 3 (in the stored definitions), in predict' in fixed


def test_fbx_agent_rewrites_line_numbers_only_when_definitions_were_prepended(tmp_path):
    game = FakeGame(tmp_path)
    gate = WorldModelGate(probe_actions=8, max_refusals=8, mechanisms="all-nocont-fbx")
    with FakeOpenAIServer([tool_call_response("x = 1\ny = undefined_name"), error_response(503, "s")]) as server:
        _analyze(_agent(server, gate, "all-nocont-fbx"), game, 1)
        tools = [json.loads(m["content"]) for m in server.requests[1]["messages"] if m["role"] == "tool"]
    assert "NameError" in tools[-1]["error"] and "stored definitions" not in tools[-1]["error"]


def test_capture_keeps_only_self_contained_constants_in_fbx():
    code = ("import math\ncorrect = 3\nfor i in range(3):\n    total = i\nSCALE = 2\nRATIO = correct / total\n"
            "LIMIT = SCALE * math.pi\nSQUARES = [k * k for k in (1, 2)]\n"
            "def predict(ascii, action):\n    return ascii\n")
    old = extract_definitions(code)
    new = extract_definitions(code, self_contained=True)
    assert "const:RATIO" in old and "const:RATIO" not in new  # `total` comes from a dropped loop
    for kept in ("const:correct", "const:SCALE", "const:LIMIT", "const:SQUARES", "predict"):
        assert kept in new
    assert "const:RATIO" in extract_definitions(code, self_contained=True, known={"total"})


def test_stored_names_cover_defs_imports_and_constants():
    store = {"predict": "def predict(a, b):\n    return a", "import:import numpy as np": "import numpy as np",
             "const:K": "K = 3"}
    assert stored_names(store) >= {"predict", "np", "K"}


def test_fbx_is_a_valid_build_value(tmp_path):
    from wmagent.kaggle.build import main

    main(["--out", str(tmp_path), "--label", "t", "--wm-mechanisms", "all-nocont-fbx", "--dry-run"])
    env = json.loads((tmp_path / "kaggle" / "source-dataset" / "agent_env.json").read_text())
    assert env["WM_MECHANISMS"] == "all-nocont-fbx"
    gate = WorldModelGate(probe_actions=8, max_refusals=8, mechanisms="all-nocont-fbx")
    assert gate.full and gate.capture_feedback and gate.fixes
