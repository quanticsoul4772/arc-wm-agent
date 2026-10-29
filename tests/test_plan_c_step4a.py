"""Plan C step 4a (WM_MECHANISMS=all): deny evidence (M1b), carried probation grant (M1c), release valve (M1d).
The base and m1a arms keep the old gate behavior."""
import json

from wmagent.world_model_gate import WorldModelGate

PREDICT = "def predict(ascii, action):\n    return ascii\n"


def _gate(tmp_path, mechanisms="all", probe_actions=1, max_refusals=8):
    gate = WorldModelGate(probe_actions=probe_actions, max_refusals=max_refusals, mechanisms=mechanisms)
    tmp_path.mkdir(parents=True, exist_ok=True)
    gate.reset(tmp_path / "g_p0_tool_runtime_state.json")
    gate.observe_level(1)
    gate.observe_class(["UP", "DOWN"])
    return gate


def _events(gate):
    return [json.loads(x) for x in gate.log_path.read_text().splitlines()]


def _rows(level, outcomes, start=0):
    """Backtest rows [index, changed, miss, error, level, action]: True = hit, False = miss (5 cells changed)."""
    return [[start + i, 5, 0 if hit else 5, None, level, "UP"] for i, hit in enumerate(outcomes)]


# --- M1b ------------------------------------------------------------------


def test_all_arm_refusal_carries_the_levels_last_verdict(tmp_path):
    gate = _gate(tmp_path)
    gate.after_cell(PREDICT, False, lambda src: _rows(1, [True, False, False]))
    gate.record_actions(1)
    refusal = gate.check_action(1)
    assert "Your last predict matched 1 of 3 informative moves" in refusal
    assert "First miss: UP changed 5 cells and predict was off by 5 cells." in refusal
    assert len(gate.evidence_line()) <= 250


def test_evidence_before_any_judged_predict_and_in_other_arms(tmp_path):
    gate = _gate(tmp_path)
    gate.record_actions(1)
    assert "No predict has been judged on this level yet" in gate.check_action(1)
    for arm in ("base", "m1a"):
        other = _gate(tmp_path / arm, mechanisms=arm)
        other.record_actions(1)
        assert "No predict has been judged" not in other.check_action(1)


def test_blocked_note_is_given_only_while_blocked_on_the_current_level(tmp_path):
    gate = _gate(tmp_path)
    assert gate.blocked_note(1) is None  # nothing refused yet
    gate.record_actions(1)
    gate.check_action(1)
    assert gate.blocked_note(1).startswith("World-model gate: action() is blocked on this level.")
    assert gate.blocked_note(2) is None  # the board moved on; the gate sees it at the next action()
    base = _gate(tmp_path / "b", mechanisms="base")
    base.record_actions(1)
    base.check_action(1)
    assert base.blocked_note(1) is None


def test_deny_followup_records_whether_the_next_predict_changed_and_improved(tmp_path):
    gate = _gate(tmp_path)
    gate.after_cell(PREDICT, False, lambda src: _rows(1, [True, False, False]))
    gate.record_actions(1)
    gate.check_action(1)
    better = "def predict(ascii, action):\n    return ascii + ''\n"
    gate.after_cell(better, False, lambda src: _rows(1, [True, True, False]))
    follow = [e for e in _events(gate) if e["event"] == "deny_followup"]
    assert follow[-1]["predict_changed"] is True
    assert follow[-1]["accuracy_before"] == 0.333 and follow[-1]["accuracy_after"] == 0.667
    gate.check_action(1)
    gate.after_cell(better, False, lambda src: _rows(1, [True, True, False]))  # the same code resubmitted
    follow = [e for e in _events(gate) if e["event"] == "deny_followup"]
    assert len(follow) == 2 and follow[-1]["predict_changed"] is False


# --- M1c ------------------------------------------------------------------


def test_confirmed_model_carries_a_grant_into_the_next_level_only(tmp_path):
    gate = _gate(tmp_path)
    gate.store = {"predict": PREDICT}
    gate.confirmed = True
    gate.observe_level(2)
    assert gate.grant and "grant" in [e["event"] for e in _events(gate)]
    gate.record_actions(5)
    assert gate.check_action(3) is None  # past the probe budget, no refusal while the grant holds
    gate.observe_level(3)  # no confirmation on level 2: no grant on level 3
    assert not gate.grant
    gate.record_actions(1)
    assert gate.check_action(1) is not None


def test_two_consecutive_misses_revoke_the_grant_and_the_gate_resumes(tmp_path):
    gate = _gate(tmp_path)
    gate.store = {"predict": PREDICT}
    gate.confirmed = True
    gate.observe_level(2)
    gate.record_actions(4)
    gate.after_cell("x = 1", False, lambda src: _rows(2, [True, False, True, False], start=10))
    assert gate.grant  # misses were not consecutive
    gate.after_cell("x = 1", False, lambda src: _rows(2, [True, False, True, False, False], start=10))
    assert not gate.grant and "grant_revoked" in [e["event"] for e in _events(gate)]
    assert gate.check_action(1) is not None  # 4 actions spent on the level, budget 1


def test_no_grant_outside_the_all_arm(tmp_path):
    for arm in ("base", "m1a"):
        gate = _gate(tmp_path / arm, mechanisms=arm)
        gate.confirmed = True
        gate.observe_level(2)
        assert not gate.grant


# --- M1d ------------------------------------------------------------------


def test_release_valve_after_eight_calls_on_a_blocked_level(tmp_path):
    gate = _gate(tmp_path)
    gate.note_model_call()
    assert gate.blocked_calls == 0  # not blocked yet
    gate.record_actions(1)
    gate.check_action(1)
    for _ in range(7):
        gate.note_model_call()
    assert not gate.released and gate.check_action(1) is not None
    gate.note_model_call()
    assert gate.released and gate.check_action(1) is None
    released = next(e for e in _events(gate) if e["event"] == "released_blocked_calls")
    assert released["calls"] == 8 and released["refusals"] == 2
    gate.observe_level(2)
    assert gate.blocked_calls == 0 and not gate.released


def test_refusal_cap_still_releases_first_when_it_comes_first(tmp_path):
    gate = _gate(tmp_path, max_refusals=2)
    gate.record_actions(1)
    assert gate.check_action(1) is not None
    assert gate.check_action(1) is not None
    assert gate.check_action(1) is None and gate.released
    assert "released_sim_free" in [e["event"] for e in _events(gate)]


def test_release_valve_is_off_in_base_and_m1a(tmp_path):
    for arm in ("base", "m1a"):
        gate = _gate(tmp_path / arm, mechanisms=arm)
        gate.record_actions(1)
        gate.check_action(1)
        for _ in range(20):
            gate.note_model_call()
        assert not gate.released


# --- agent wiring -----------------------------------------------------------


def test_agent_carries_the_blocked_note_and_counts_calls(tmp_path):
    from dataclasses import replace

    from tests.fake_openai import FakeOpenAIServer, error_response, tool_call_response
    from tests.test_agent import FakeGame, FakeRefused, FakeSandbox, _analyze
    from wmagent.agent import CONFIG, ToolAgent

    game = FakeGame(tmp_path)
    gate = WorldModelGate(probe_actions=1, max_refusals=8, mechanisms="all")
    script = [tool_call_response('action("UP")'), tool_call_response('action("UP")'),
              tool_call_response('print("thinking")'), error_response(503, "stop")]
    with FakeOpenAIServer(script) as server:
        config = replace(CONFIG, base_url=server.base_url, mechanisms="all")
        agent = ToolAgent(config=config, gate=gate, sandbox=(FakeSandbox(), FakeRefused))
        assert _analyze(agent, game, 1)["step_executed"]
        _analyze(agent, game, 2)  # refused, then a no-action cell, then the server stops
        texts = []
        for request in server.requests:
            user = [m for m in request["messages"] if m["role"] == "user"][-1]
            texts.append(user["content"][0]["text"] if isinstance(user["content"], list) else user["content"])
    assert "action() is blocked on this level" not in texts[1]  # nothing refused when turn 2 began
    assert gate.blocked() and gate.blocked_calls >= 1
