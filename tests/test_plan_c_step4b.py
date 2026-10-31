"""Plan C step 4b (WM_MECHANISMS=all): shadow no-op deny (M2a), STUCK line (M2b), image elision (M4b), batch
truncation and the probe line (M4c), harness-written level memory (M4d)."""
import json
from dataclasses import replace

from tests.fake_openai import FakeOpenAIServer, error_response, tool_call_response
from tests.test_agent import FakeGame, FakeRefused, FakeSandbox, _analyze
from wmagent.agent import BOARD_IMAGE_OMITTED, CONFIG, ToolAgent
from wmagent.progress import ProgressMonitor, action_kind
from wmagent.world_model_gate import WorldModelGate


def _grid(mark):
    return [[1 if (r, c) == mark else 0 for c in range(4)] for r in range(4)]


def _frame(mark, level=1):
    return {"grid": _grid(mark), "level": level}


def _monitor(tmp_path):
    return ProgressMonitor.for_state(tmp_path / "g_p0_tool_runtime_state.json")


def _events(monitor):
    if not monitor.log_path.exists():
        return []
    return [json.loads(x) for x in monitor.log_path.read_text().splitlines()]


def _agent(server, gate=None, mechanisms="all"):
    config = replace(CONFIG, base_url=server.base_url, mechanisms=mechanisms)
    return ToolAgent(config=config, gate=gate, sandbox=(FakeSandbox(), FakeRefused))


def _last_user_text(request):
    user = [m for m in request["messages"] if m["role"] == "user"][-1]
    return user["content"][0]["text"] if isinstance(user["content"], list) else user["content"]


# --- M2a shadow no-op deny ------------------------------------------------------


def test_noop_pairs_are_flagged_in_shadow_and_their_outcome_logged(tmp_path):
    m = _monitor(tmp_path)
    m.record([_frame((0, 0)), _frame((0, 0))], ["UP"], [])  # UP changed nothing on this board
    assert m.shadow_check(1, _grid((0, 0)), ["UP", "UP", "DOWN"]) == [0, 1]
    assert m.shadow_check(1, _grid((0, 0)), ["DOWN", "UP"]) == []  # only a leading run of known no-ops
    m.record([_frame((0, 0)), _frame((1, 0))], ["UP"], [0])  # hidden state: this time it moved
    events = _events(m)
    assert [e["event"] for e in events] == ["noop_would_deny", "noop_outcome"]
    assert events[0]["positions"] == [0, 1] and events[0]["actions"] == ["UP", "UP"]
    assert events[-1] == {**events[-1], "event": "noop_outcome", "action": "UP", "changed": True}
    assert m.shadow_check(1, _grid((0, 0)), ["UP"]) == []  # the pair changed the board, so it left the table


def test_noop_table_clears_on_reset_and_level_change(tmp_path):
    m = _monitor(tmp_path)
    m.record([_frame((0, 0)), _frame((0, 0))], ["UP"], [])
    m.record([_frame((0, 0)), _frame((0, 0))], ["RESET"], [])
    assert m.shadow_check(1, _grid((0, 0)), ["UP"]) == []
    m.record([_frame((0, 0)), _frame((0, 0))], ["UP"], [])
    assert m.shadow_check(2, _grid((0, 0)), ["UP"]) == []


# --- M2b STUCK -------------------------------------------------------------------


def test_two_no_change_actions_give_one_stuck_line_and_a_followup(tmp_path):
    m = _monitor(tmp_path)
    m.record([_frame((0, 0)), _frame((0, 0)), _frame((0, 0))], ["UP", "UP"], [])
    line = m.take_stuck_line()
    assert line.startswith("Progress check: Your last 2 actions changed nothing") and "UPx2" in line
    assert m.take_stuck_line() is None
    m.record([_frame((0, 0)), _frame((1, 0)), _frame((2, 0)), _frame((3, 0))], ["DOWN", "DOWN", "DOWN"], [])
    follow = next(e for e in _events(m) if e["event"] == "stuck_followup")
    assert follow["differed"] is True and follow["next_actions"] == ["DOWN", "DOWN", "DOWN"]


def test_repeated_cell_and_uncaptured_refusals_are_stuck_signals(tmp_path):
    m = _monitor(tmp_path)
    m.record_cell("print(1)")
    m.record_cell("print(1)\n")
    assert m.take_stuck_line() is None
    m.record_cell("print(1)")
    assert "same python cell 3 times" in m.take_stuck_line()
    m.record_refusal(captured=False)
    m.record_refusal(captured=True)
    m.record_refusal(captured=False)
    assert m.take_stuck_line() is None
    m.record_refusal(captured=False)
    assert "refused 2 times in a row" in m.take_stuck_line()


# --- M4d level memory -------------------------------------------------------------


def test_level_memory_is_recorded_rendered_and_its_use_measured(tmp_path):
    m = _monitor(tmp_path)
    frames = [_frame((0, 0)), _frame((1, 0)), _frame((1, 0)), _frame((2, 0), level=2)]
    m.record(frames, ["DOWN", "LEFT", "DOWN"], [])
    block = m.memory_block()
    assert "Level 1: 3 actions (DOWN, LEFT, DOWN)" in block and "DOWN 100%" in block and "LEFT 0%" in block
    assert "No effect observed: LEFT (1 level)." in block
    level2 = [_frame((2, 0), 2), _frame((3, 0), 2), _frame((3, 0), 2), _frame((3, 0), 2), _frame((2, 0), 2),
              _frame((1, 0), 3)]
    m.record(level2, ["DOWN", "LEFT", "LEFT", "UP", "UP"], [])
    assert "No effect observed: LEFT (2 levels)." in m.memory_block()
    use = next(e for e in _events(m) if e["event"] == "memory_use")
    assert use["first_actions"] == ["DOWN", "LEFT", "LEFT", "UP", "UP"] and use["used_effective"] == 1


def test_action_kind_drops_click_coordinates():
    assert action_kind("MOUSE(row=3, col=4)") == "MOUSE" and action_kind("UP") == "UP"


# --- agent wiring -----------------------------------------------------------------


def test_agent_stuck_line_and_shadow_noop_from_real_actions(tmp_path):
    game = FakeGame(tmp_path)  # marker at row 4; UP stops at row 0
    script = [tool_call_response('action(["UP"] * 6)'), tool_call_response('action("UP")'),
              tool_call_response('action("DOWN")')]
    with FakeOpenAIServer(script) as server:
        agent = _agent(server)
        for step in (1, 2, 3):
            _analyze(agent, game, step)
        texts = [_last_user_text(r) for r in server.requests]
    assert "Progress check: Your last 2 actions changed nothing" in texts[1]
    events = [json.loads(x) for x in (game.state_path.parent / "g1_p0_progress.jsonl").read_text().splitlines()]
    names = [e["event"] for e in events]
    assert "noop_would_deny" in names and {"event": "noop_outcome", "changed": False}.items() <= next(
        e for e in events if e["event"] == "noop_outcome").items()
    assert game.history[-2]["action"] == "UP"  # shadow only: the no-op UP still ran


def test_kept_user_messages_lose_the_image_only_in_the_all_arm(tmp_path):
    for arm, expect_image in (("all", False), ("base", True)):
        game = FakeGame(tmp_path / arm)
        script = [tool_call_response('action("UP")'), tool_call_response('action("UP")')]
        with FakeOpenAIServer(script) as server:
            agent = _agent(server, mechanisms=arm)
            _analyze(agent, game, 1)
            kept = [m for m in agent.history if m["role"] == "user"]
            _analyze(agent, game, 2)
            second = server.requests[1]["messages"]
        has_image = any(p.get("type") == "image_url" for m in kept for p in m["content"])
        assert has_image is expect_image
        live = [m for m in second if m["role"] == "user"][-1]
        assert any(p.get("type") == "image_url" for p in live["content"])  # the live board always rides along
        if arm == "all":
            assert any(p.get("text") == BOARD_IMAGE_OMITTED for m in kept for p in m["content"])
            assert kept == [m for m in agent.history if m["role"] == "user"][:len(kept)]  # unchanged once kept


def test_oversized_probe_batch_is_truncated_in_all_and_capped_in_base(tmp_path):
    for arm in ("all", "base"):
        game = FakeGame(tmp_path / arm)
        gate = WorldModelGate(probe_actions=2, max_refusals=8, mechanisms=arm)
        with FakeOpenAIServer([tool_call_response('action(["DOWN"] * 4)'), error_response(503, "x")]) as server:
            agent = _agent(server, gate=gate, mechanisms=arm)
            _analyze(agent, game, 1)
            tool = [json.loads(m["content"]) for m in agent.history if m["role"] == "tool"]
        events = [json.loads(x)["event"] for x in gate.log_path.read_text().splitlines()]
        if arm == "all":
            assert game.action_num == 2 and "batch_truncated" in events
            assert "only the first 2 action(s) ran" in tool[-1]["result"]["truncated_by_gate"]
        else:
            assert game.action_num == 0 and "batch_capped" in events


def test_probe_line_and_memory_block_reach_the_prompt(tmp_path):
    game = FakeGame(tmp_path, level_len=2)
    gate = WorldModelGate(probe_actions=8, max_refusals=8, mechanisms="all")
    script = [tool_call_response('action(["DOWN", "DOWN"])'), tool_call_response('action("DOWN")'),
              error_response(503, "x")]
    with FakeOpenAIServer(script) as server:
        agent = _agent(server, gate=gate)
        _analyze(agent, game, 1)
        _analyze(agent, game, 2)
        texts = [_last_user_text(r) for r in server.requests]
    assert "8 probe action(s) left on this level" in texts[0]
    assert "Solved levels (recorded by the harness):" in texts[1] and "Level 1: 2 actions (DOWN, DOWN)" in texts[1]
    assert "8 probe action(s) left on this level" in texts[1]  # the new level's budget
