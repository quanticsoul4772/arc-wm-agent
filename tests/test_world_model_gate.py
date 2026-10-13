import json
from pathlib import Path

from inference.agent.python_tool_sandbox import ActionRefused, run_sandboxed_python
from inference.agent.world_model_gate import (
    BACKTEST_DRIVER,
    WorldModelGate,
    cell_mismatch,
    extract_definitions,
)


def _board(fill_row: int, size: int = 8) -> str:
    return "\n".join(("x" * size) if r == fill_row else ("." * size) for r in range(size))


def _state(boards: list[str], actions: list[str], level: int = 1) -> dict:
    history = []
    for step, (board, action) in enumerate(zip(boards, actions)):
        history.append(
            {
                "action": action,
                "frame": {"ascii": board, "step": step, "level": level, "shape": [8, 8], "grid": []},
            }
        )
    return {"current_frame": history[-1]["frame"], "history": history, "valid_actions": ["DOWN"], "last_action_result": {}}


def _run_backtest(source: str, state: dict):
    def _no_actions(_):
        raise ActionRefused("no actions")

    outcome = run_sandboxed_python(
        code=source + "\n\n" + BACKTEST_DRIVER,
        timeout_seconds=10,
        initial_state=state,
        action_handler=_no_actions,
    )
    assert not outcome.get("error"), outcome
    return outcome["result"]


CORRECT_PREDICT = '''
def shift(rows):
    return [rows[-1]] + rows[:-1]

def predict(ascii, action):
    rows = ascii.splitlines()
    return "\\n".join(shift(rows)) if action == "DOWN" else ascii

action(["DOWN"])
'''


def test_extract_requires_top_level_predict():
    assert extract_definitions("x = 1\nprint(x)") is None
    assert extract_definitions("def helper():\n    def predict(a, b):\n        return a\n") is None
    assert extract_definitions('NOTE = """\ndef predict(a, b):\n"""\n') is None


def test_extract_keeps_definitions_and_drops_calls():
    defs = extract_definitions(CORRECT_PREDICT)
    assert set(defs) == {"shift", "predict"}
    assert "action(" not in "\n".join(defs.values())


def test_cell_mismatch_counts_cells():
    assert cell_mismatch(_board(0), _board(0)) == 0
    assert cell_mismatch(_board(0), _board(1)) == 16


def test_backtest_and_judge_accept_correct_model(tmp_path: Path):
    boards = [_board(r) for r in range(5)]
    state = _state(boards, ["RESET", "DOWN", "DOWN", "DOWN", "DOWN"])
    gate = WorldModelGate(probe_actions=2, max_refusals=2)
    gate.reset(tmp_path / "g_p0_tool_runtime_state.json")
    gate.observe_level(1)
    gate.after_cell(CORRECT_PREDICT, False, lambda src: _run_backtest(src, state))
    assert gate.accepted_at is not None
    events = [json.loads(line)["event"] for line in (tmp_path / "g_p0_wm_gate.jsonl").read_text().splitlines()]
    assert "accepted" in events


def test_identity_model_is_rejected(tmp_path: Path):
    boards = [_board(r) for r in range(5)]
    state = _state(boards, ["RESET", "DOWN", "DOWN", "DOWN", "DOWN"])
    gate = WorldModelGate(probe_actions=2, max_refusals=2)
    gate.reset(tmp_path / "g_p0_tool_runtime_state.json")
    gate.after_cell("def predict(ascii, action):\n    return ascii\n", False, lambda src: _run_backtest(src, state))
    assert gate.accepted_at is None


def test_gate_refuses_then_releases(tmp_path: Path):
    gate = WorldModelGate(probe_actions=2, max_refusals=2)
    gate.reset(tmp_path / "g_p0_tool_runtime_state.json")
    gate.observe_level(1)
    assert gate.check_action(1) is None
    gate.record_actions(2)
    assert gate.check_action(1) is not None
    assert gate.check_action(1) is not None
    assert gate.check_action(1) is None
    assert gate.released
    gate.observe_level(2)
    assert not gate.released and gate.level_actions == 0


def test_prepend_makes_predict_available():
    gate = WorldModelGate(probe_actions=2, max_refusals=2)
    gate.store = extract_definitions(CORRECT_PREDICT)
    code = gate.prepend("result = predict('a\\nb', 'DOWN')")

    def _no_actions(_):
        raise ActionRefused("no actions")

    outcome = run_sandboxed_python(
        code=code, timeout_seconds=10, initial_state=_state([_board(0)], ["RESET"]), action_handler=_no_actions
    )
    assert outcome["result"] == "b\na"


def _small_board(cells_on: int, size: int = 8) -> str:
    flat = ["x" if i < cells_on else "." for i in range(size * size)]
    return "\n".join("".join(flat[r * size:(r + 1) * size]) for r in range(size))


SMALL_PREDICT = '''
def predict(ascii, action):
    flat = ascii.replace("\\n", "")
    on = flat.count("x") + 3
    cells = ["x" if i < on else "." for i in range(64)]
    return "\\n".join("".join(cells[r * 8:(r + 1) * 8]) for r in range(8))
'''


def test_small_changes_are_informative(tmp_path: Path):
    boards = [_small_board(3 * k) for k in range(5)]
    state = _state(boards, ["RESET", "CLICK", "CLICK", "CLICK", "CLICK"])
    accept = WorldModelGate(probe_actions=2, max_refusals=2)
    accept.reset(tmp_path / "a_p0_tool_runtime_state.json")
    accept.after_cell(SMALL_PREDICT, False, lambda src: _run_backtest(src, state))
    assert accept.accepted_at is not None
    identity = WorldModelGate(probe_actions=2, max_refusals=2)
    identity.reset(tmp_path / "b_p0_tool_runtime_state.json")
    identity.after_cell("def predict(ascii, action):\n    return ascii\n", False, lambda src: _run_backtest(src, state))
    assert identity.accepted_at is None


def test_skeleton_is_seeded_and_cannot_pass(tmp_path: Path):
    from inference.agent.world_model_gate import SKELETON_PREDICT

    gate = WorldModelGate(probe_actions=2, max_refusals=2)
    gate.reset(tmp_path / "s_p0_tool_runtime_state.json")
    assert gate.store == {"predict": SKELETON_PREDICT}
    boards = [_board(r) for r in range(5)]
    state = _state(boards, ["RESET", "DOWN", "DOWN", "DOWN", "DOWN"])
    gate.observe_level(1)
    gate.after_cell("x = 1\n", False, lambda src: _run_backtest(src, state))
    assert gate.accepted_at is None


def test_batch_crossing_the_probe_budget_is_capped(tmp_path: Path):
    gate = WorldModelGate(probe_actions=5, max_refusals=2)
    gate.reset(tmp_path / "b_p0_tool_runtime_state.json")
    gate.observe_level(1)
    gate.record_actions(3)
    assert gate.check_action(2) is None
    message = gate.check_action(100)
    assert message is not None and "at most 2" in message
    assert gate.refusals == 0


def test_reset_and_pre_reset_transitions_are_skipped():
    boards = [_board(0), _board(1), _board(5), _board(0), _board(1)]
    state = _state(boards, ["RESET", "DOWN", "LEFT", "RESET", "DOWN"])
    rows = _run_backtest(CORRECT_PREDICT_NO_ACTION, state)
    indices = [r[0] for r in rows]
    assert 3 not in indices  # the RESET transition
    assert 2 not in indices  # the transition just before a RESET
    assert indices == [1, 4]


CORRECT_PREDICT_NO_ACTION = CORRECT_PREDICT.replace('action(["DOWN"])', "")


def test_level_change_clears_acceptance_and_carried_model_is_rejudged(tmp_path: Path):
    boards = [_board(r) for r in range(5)]
    state = _state(boards, ["RESET", "DOWN", "DOWN", "DOWN", "DOWN"])
    gate = WorldModelGate(probe_actions=2, max_refusals=2)
    gate.reset(tmp_path / "l_p0_tool_runtime_state.json")
    gate.observe_level(1)
    gate.after_cell(CORRECT_PREDICT, False, lambda src: _run_backtest(src, state))
    assert gate.accepted_at is not None
    gate.observe_level(2)
    assert gate.accepted_at is None and gate.prequential == []
    level2 = _state(boards, ["RESET", "DOWN", "DOWN", "DOWN", "DOWN"], level=2)
    gate.after_cell("y = 2\n", False, lambda src: _run_backtest(src, level2))
    assert gate.accepted_at is not None
    events = [json.loads(line) for line in (tmp_path / "l_p0_wm_gate.jsonl").read_text().splitlines()]
    assert any(e["event"] == "accepted" and e.get("carried") for e in events)


def test_backtest_error_is_logged(tmp_path: Path):
    gate = WorldModelGate(probe_actions=2, max_refusals=2)
    gate.reset(tmp_path / "e_p0_tool_runtime_state.json")
    gate.note_backtest_error("Sandbox timeout")
    events = [json.loads(line) for line in (tmp_path / "e_p0_wm_gate.jsonl").read_text().splitlines()]
    assert events[-1]["event"] == "backtest_error"


def test_kaggle_setup_script_carries_probe_settings(monkeypatch):
    from inference.framework.kaggle import duck_kaggle_setup_command

    monkeypatch.setenv("PROBE_ENFORCE", "1")
    monkeypatch.setenv("PROBE_ACTIONS", "15")
    script = duck_kaggle_setup_command()
    assert "'PROBE_ENFORCE': '1'" in script
    assert "'PROBE_ACTIONS': '15'" in script
    assert "'PROBE_MAX_REFUSALS': '8'" in script


def test_kaggle_setup_script_defaults_probe_off(monkeypatch):
    from inference.framework.kaggle import duck_kaggle_setup_command

    monkeypatch.delenv("PROBE_ENFORCE", raising=False)
    script = duck_kaggle_setup_command()
    assert "'PROBE_ENFORCE': '0'" in script


MEMORIZING_PREDICT = '''
SEEN = {BOARDS}

def predict(ascii, action):
    return SEEN.get(ascii, ascii)
'''


def _heldout_events(path: Path) -> list[dict]:
    return [e for e in (json.loads(line) for line in path.read_text().splitlines()) if e["event"] == "heldout"]


def _accept_then_play_on(tmp_path: Path, source: str, name: str) -> list[dict]:
    """Accept on boards 0..4, then score transitions 5..7 the model never saw."""
    boards = [_board(r) for r in range(8)]
    seen = _state(boards[:5], ["RESET"] + ["DOWN"] * 4)
    later = _state(boards, ["RESET"] + ["DOWN"] * 7)
    gate = WorldModelGate(probe_actions=2, max_refusals=2)
    gate.reset(tmp_path / f"{name}_p0_tool_runtime_state.json")
    gate.observe_level(1)
    gate.after_cell(source, False, lambda src: _run_backtest(src, seen))
    assert gate.accepted_at is not None
    gate.after_cell("z = 3\n", False, lambda src: _run_backtest(src, later))
    return _heldout_events(tmp_path / f"{name}_p0_wm_gate.jsonl")


def test_correct_model_is_confirmed_on_heldout(tmp_path: Path):
    events = _accept_then_play_on(tmp_path, CORRECT_PREDICT_NO_ACTION, "c")
    assert len(events) == 1
    assert events[0]["confirmed"] and not events[0]["false_confidence"]
    assert events[0]["heldout_hits"] == 3


def test_memorizing_model_fails_heldout_with_false_confidence(tmp_path: Path):
    seen = {_board(r): _board(r + 1) for r in range(4)}
    source = MEMORIZING_PREDICT.replace("{BOARDS}", repr(seen))
    events = _accept_then_play_on(tmp_path, source, "m")
    assert len(events) == 1
    assert not events[0]["confirmed"] and events[0]["false_confidence"]
    assert events[0]["heldout_hits"] == 0


def test_heldout_needs_enough_followups_and_does_not_gate(tmp_path: Path):
    boards = [_board(r) for r in range(7)]
    gate = WorldModelGate(probe_actions=2, max_refusals=2)
    gate.reset(tmp_path / "h_p0_tool_runtime_state.json")
    gate.observe_level(1)
    gate.after_cell(CORRECT_PREDICT, False, lambda src: _run_backtest(src, _state(boards[:5], ["RESET"] + ["DOWN"] * 4)))
    gate.after_cell("z = 3\n", False, lambda src: _run_backtest(src, _state(boards, ["RESET"] + ["DOWN"] * 6)))
    assert _heldout_events(tmp_path / "h_p0_wm_gate.jsonl") == []
    gate.record_actions(5)
    assert gate.check_action(1) is None


def _accept(gate: WorldModelGate, tmp_path: Path, name: str) -> None:
    boards = [_board(r) for r in range(5)]
    gate.reset(tmp_path / f"{name}_p0_tool_runtime_state.json")
    gate.observe_level(1)
    gate.after_cell(CORRECT_PREDICT, False, lambda src: _run_backtest(src, _state(boards, ["RESET"] + ["DOWN"] * 4)))
    assert gate.accepted_at is not None


def test_release_after_accept_stops_refusals_on_later_levels(tmp_path: Path):
    gate = WorldModelGate(probe_actions=2, max_refusals=2, release_after_accept=True)
    _accept(gate, tmp_path, "r")
    gate.observe_level(2)
    gate.record_actions(5)
    assert gate.check_action(1) is None
    assert gate.refusals == 0


def test_default_gate_still_refuses_on_a_new_level(tmp_path: Path):
    gate = WorldModelGate(probe_actions=2, max_refusals=2)
    _accept(gate, tmp_path, "d")
    gate.observe_level(2)
    gate.record_actions(5)
    assert gate.check_action(1) is not None


def test_kaggle_setup_script_carries_release_setting(monkeypatch):
    from inference.framework.kaggle import duck_kaggle_setup_command

    monkeypatch.setenv("PROBE_RELEASE_AFTER_ACCEPT", "1")
    assert "'PROBE_RELEASE_AFTER_ACCEPT': '1'" in duck_kaggle_setup_command()
    monkeypatch.delenv("PROBE_RELEASE_AFTER_ACCEPT")
    assert "'PROBE_RELEASE_AFTER_ACCEPT': '0'" in duck_kaggle_setup_command()


def _optional_gate(tmp_path: Path, name: str) -> WorldModelGate:
    gate = WorldModelGate(probe_actions=2, max_refusals=2, optional=True)
    gate.reset(tmp_path / f"{name}_p0_tool_runtime_state.json")
    gate.observe_level(1)
    return gate


def _run_cell(gate: WorldModelGate, code: str, state: dict):
    def _no_actions(_):
        raise ActionRefused("no actions")

    outcome = run_sandboxed_python(
        code=gate.prepend(code), timeout_seconds=10, initial_state=state, action_handler=_no_actions
    )
    assert not outcome.get("error"), outcome
    return outcome["result"]


def test_optional_mode_never_refuses(tmp_path: Path):
    gate = _optional_gate(tmp_path, "o")
    gate.record_actions(50)
    assert gate.check_action(10) is None
    assert gate.refusals == 0


def test_simulate_is_blocked_until_heldout_confirmation(tmp_path: Path):
    boards = [_board(r) for r in range(8)]
    seen = _state(boards[:5], ["RESET"] + ["DOWN"] * 4)
    later = _state(boards, ["RESET"] + ["DOWN"] * 7)
    gate = _optional_gate(tmp_path, "s")
    gate.after_cell(CORRECT_PREDICT_NO_ACTION, False, lambda src: _run_backtest(src, seen))
    assert gate.accepted_at is not None and not gate.confirmed
    blocked = _run_cell(gate, "result = simulate(['DOWN'])", seen)
    assert isinstance(blocked, str) and "not confirmed" in blocked
    gate.after_cell("z = 3\n", False, lambda src: _run_backtest(src, later))
    assert gate.confirmed
    boards_out = _run_cell(gate, "result = simulate(['DOWN', 'DOWN'])", later)
    assert boards_out == [_board(0), _board(1)]


def test_confirmation_resets_on_level_change(tmp_path: Path):
    boards = [_board(r) for r in range(8)]
    gate = _optional_gate(tmp_path, "c")
    gate.after_cell(CORRECT_PREDICT_NO_ACTION, False, lambda src: _run_backtest(src, _state(boards[:5], ["RESET"] + ["DOWN"] * 4)))
    gate.after_cell("z = 3\n", False, lambda src: _run_backtest(src, _state(boards, ["RESET"] + ["DOWN"] * 7)))
    assert gate.confirmed
    gate.observe_level(2)
    assert not gate.confirmed


def test_kaggle_setup_script_carries_optional_setting(monkeypatch):
    from inference.framework.kaggle import duck_kaggle_setup_command

    monkeypatch.setenv("PROBE_OPTIONAL", "1")
    assert "'PROBE_OPTIONAL': '1'" in duck_kaggle_setup_command()
    monkeypatch.delenv("PROBE_OPTIONAL")
    assert "'PROBE_OPTIONAL': '0'" in duck_kaggle_setup_command()
