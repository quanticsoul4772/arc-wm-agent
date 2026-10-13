import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.fake_openai import (
    FakeOpenAIServer,
    chat_body,
    error_response,
    text_response,
    tool_call_response,
)
from wmagent import agent as agent_mod
from wmagent import prompts
from wmagent.agent import (
    CONFIG,
    ToolAgent,
    drop_oldest_block,
    estimate_tokens,
    extract_notes,
    persistable_history,
    recover_tool_calls,
    trim_messages,
)
from wmagent.world_model_gate import BACKTEST_DRIVER, WorldModelGate

# --------------------------------------------------------------------------- fakes


class FakeRefused(Exception):
    pass


class FakeSandbox:
    """In-process stand-in for wmagent.sandbox.run_sandboxed_python (test code only)."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def __call__(self, *, code, timeout_seconds, initial_state, action_handler):
        self.calls.append({"code": code, "state": initial_state, "handler": action_handler})
        out: list[str] = []
        scope: dict = {}
        action_results: list = []

        def load(state):
            frames = [_frame(e["frame"]) for e in state["history"]]
            history = [SimpleNamespace(action=e["action"], frame=f) for e, f in zip(state["history"], frames)]
            transitions = [
                SimpleNamespace(action=h.action, before_frame=frames[i - 1] if i else None, after_frame=h.frame)
                for i, h in enumerate(history)
                if h.action
            ]
            scope.update(
                current_frame=_frame(state["current_frame"]),
                history=history,
                transitions=transitions,
                valid_actions=state["valid_actions"],
                last_action_result=state["last_action_result"],
            )

        def action(x):
            try:
                reply = action_handler(x if isinstance(x, list) else [x])
            except FakeRefused as exc:
                raise RuntimeError(str(exc)) from None
            action_results.append(reply["action_result"])
            load(reply["state"])
            return reply["action_result"]

        load(initial_state)
        scope.update(action=action, result=None, print=lambda *a: out.append(" ".join(map(str, a))))
        try:
            exec(code, scope)  # noqa: S102 - test fake for the subprocess sandbox
        except Exception as exc:  # noqa: BLE001 - mirrors the sandbox's error report
            return {"error": f"{type(exc).__name__}: {exc}", "stdout": "\n".join(out), "action_results": action_results}
        return {"stdout": "\n".join(out), "result": scope.get("result"), "action_results": action_results}


def _frame(f):
    return SimpleNamespace(ascii=f["ascii"], level=f["level"], step=f["step"], shape=f["shape"])


class FakeGame:
    """Marker on an 8x8 board; UP/DOWN move it. A level completes after `level_len` actions."""

    def __init__(self, tmp_path: Path, level_len: int = 100, levels: int = 3, game_over_at: int | None = None,
                 gate: WorldModelGate | None = None):
        self.gate = gate
        self.state_path = tmp_path / "artifacts" / "g1_p0_tool_runtime_state.json"
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.row, self.level, self.levels, self.level_len = 4, 1, levels, level_len
        self.level_actions = 0
        self.action_num = 0
        self.game_over_at = game_over_at
        self.history = [{"action": "", "frame": self._frame()}]
        self.calls: list[dict] = []
        self._write()

    def _frame(self):
        grid = [[8 if (r == self.row and c == 3) else 0 for c in range(8)] for r in range(8)]
        return {"grid": grid, "step": self.action_num, "level": self.level}

    def _write(self):
        self.state_path.write_text(json.dumps({"current_frame": self.history[-1]["frame"], "history": self.history}))

    def step_env(self, arguments):
        actions = arguments["actions"]
        if self.gate is not None:  # the session's step_env owns the gate accounting
            self.gate.observe_level(self.level)
            refusal = self.gate.check_action(len(actions))
            if refusal:
                raise FakeRefused(refusal)
        self.calls.append(arguments)
        payload = self._play(actions)
        if self.gate is not None:
            self.gate.record_actions(payload.get("executed_count", 0))
            self.gate.observe_level(self.level)
        return payload

    def _play(self, actions):
        executed, total_reward, stop_reason = [], 0.0, None
        payload = {}
        for index, item in enumerate(actions):
            label = item["action"]
            if label not in ("UP", "DOWN"):
                if index == 0:
                    return {"executed": False, "error": f"invalid action {label}", "valid_actions": ["UP", "DOWN"]}
                stop_reason = "invalid_action"
                break
            self.row = max(0, min(7, self.row + (-1 if label == "UP" else 1)))
            self.action_num += 1
            self.level_actions += 1
            level_completed = self.level_actions >= self.level_len
            game_over = self.game_over_at == self.action_num
            reward = 0.0
            if level_completed:
                self.level += 1
                self.level_actions = 0
                reward = 1 / self.levels
            total_reward += reward
            self.history.append({"action": label, "frame": self._frame()})
            self._write()
            executed.append(label)
            payload = {
                "executed": True, "action_num": self.action_num, "level": self.level, "score": self.level - 1,
                "reward": reward, "state": "GAME_OVER" if game_over else "NOT_FINISHED",
                "valid_actions": ["UP", "DOWN"], "board_changed": True, "done": False,
                "level_completed": level_completed, "game_over": game_over, "run_complete": False,
                "action_name": label, "action_data": {}, "action_display": label, "batch_index": index,
                "batch_size": len(actions), "run_elapsed_seconds": 1.0, "time_remaining_seconds": 99.0,
            }
            if level_completed or game_over:
                stop_reason = "level_completed" if level_completed else "game_over"
                if index < len(actions) - 1:
                    break
                stop_reason = None
        payload.update({
            "reward": total_reward, "last_reward": payload.get("reward", 0.0), "batched": len(actions) > 1,
            "requested_count": len(actions), "executed_count": len(executed), "requested_actions": actions,
            "executed_actions": executed, "stopped_early": len(executed) < len(actions), "stop_reason": stop_reason,
        })
        return payload


def _agent(server, *, gate=None, sandbox=None, clock=None, **cfg):
    config = replace(CONFIG, base_url=server.base_url, **cfg)
    kwargs = {"clock": clock} if clock else {}
    return ToolAgent(config=config, gate=gate, sandbox=(sandbox or FakeSandbox(), FakeRefused), **kwargs)


def _analyze(agent, game, step=1, transcript=None):
    return agent.analyze(
        state_path=game.state_path, action_count=game.action_num, valid_actions=["ACTION1", "ACTION2", "RESET"],
        step_env=game.step_env, transcript_path=transcript, analysis_step=step, request_timeout=5.0,
        should_stop=lambda: False,
    )


def _roles(messages):
    return [m["role"] for m in messages]


# --------------------------------------------------------------------------- (a) mock-server behaviour


def test_structured_tool_call_executes_action_and_persists(tmp_path):
    game = FakeGame(tmp_path)
    with FakeOpenAIServer([tool_call_response('print("hi")\naction("UP")')]) as server:
        agent = _agent(server)
        out = _analyze(agent, game, transcript=tmp_path / "transcripts" / "g1_p0.txt")
    assert out == {"step_executed": True, "retryable_failure": False, "yielded_control": False, "reasoning": ""}
    assert game.calls == [{"actions": [{"action": "UP"}]}]
    body = server.requests[0]
    assert body["tools"][0]["function"]["name"] == "python" and body["tool_choice"] == "auto"
    assert body["chat_template_kwargs"] == {"enable_thinking": True} and body["stream"] is False
    assert "max_tokens" not in body and "seed" not in body
    user = body["messages"][-1]["content"]
    assert user[1]["image_url"]["url"].startswith("data:image/png;base64,")
    assert "UP, DOWN" in user[0]["text"]
    assert _roles(agent.history) == ["user", "assistant", "tool"]
    tool = json.loads(agent.history[-1]["content"])
    assert tool["stdout"] == "hi" and tool["action_result"]["action_display"] == "UP"
    assert "requested_count" not in tool["action_result"]
    assert agent.generated_tokens == 10
    text = (tmp_path / "transcripts" / "g1_p0.txt").read_text()
    for section in ("SYSTEM PROMPT", "USER PROMPT", "RESPONSE META", "TOOL CALL", "TOOL RESULT", "STATUS"):
        assert f"===== {section} =====" in text


def test_tool_call_markup_in_content_is_recovered(tmp_path):
    game = FakeGame(tmp_path)
    content = (
        "Moving up now.\n<tool_call>\n<function=python>\n<parameter=code>\naction(\"UP\")\n"
        "</parameter>\n</function>\n</tool_call>"
    )
    with FakeOpenAIServer([text_response(content, reasoning="<think>plan</think>\n\nmove")]) as server:
        agent = _agent(server)
        out = _analyze(agent, game)
    assert out["step_executed"] and out["reasoning"] == "plan\nmove"
    assistant = agent.history[1]
    assert assistant["content"] == "Moving up now."
    assert assistant["reasoning"] == "plan\nmove"
    assert json.loads(assistant["tool_calls"][0]["function"]["arguments"]) == {"code": 'action("UP")'}
    assert agent.history[2]["tool_call_id"] == assistant["tool_calls"][0]["id"]


def test_recover_tool_calls_parsers():
    xml = "<function=python><parameter=code>\nx = 1\n</parameter></function>"
    assert recover_tool_calls(xml) == [{"name": "python", "arguments": {"code": "x = 1"}}]
    js = '<tool_call>{"name": "python", "arguments": {"code": "y"}}</tool_call>'
    assert recover_tool_calls(js) == [{"name": "python", "arguments": {"code": "y"}}]
    assert recover_tool_calls("<tool_call> nonsense") == []


def test_no_tool_call_gets_corrective_prompt(tmp_path):
    game = FakeGame(tmp_path)
    script = [text_response("Let me think about the board."), tool_call_response('action("DOWN")')]
    with FakeOpenAIServer(script) as server:
        agent = _agent(server)
        out = _analyze(agent, game)
    assert out["step_executed"]
    second = server.requests[1]["messages"]
    assert _roles(second)[-3:] == ["user", "assistant", "user"]
    assert second[-2]["content"] == "Let me think about the board."
    assert second[-1]["content"] == prompts.corrective_prompt(False)


def test_unrecoverable_markup_gets_stricter_corrective(tmp_path):
    game = FakeGame(tmp_path)
    script = [text_response("<tool_call> broken"), tool_call_response('action("UP")')]
    with FakeOpenAIServer(script) as server:
        _analyze(_agent(server), game)
    last = server.requests[1]["messages"][-1]["content"]
    assert last == prompts.corrective_prompt(True) and last != prompts.corrective_prompt(False)


def test_context_length_error_trims_and_retries(tmp_path):
    game = FakeGame(tmp_path)
    context_error = error_response(400, {"error": {"message": "This model's maximum context length is 65536"}})
    with FakeOpenAIServer([context_error, tool_call_response('action("UP")')]) as server:
        agent = _agent(server)
        agent._state_path = game.state_path  # same game: keep the seeded history
        agent.history = [
            {"role": "user", "content": "old turn"},
            {"role": "assistant", "content": "old reply"},
        ]
        out = _analyze(agent, game)
    assert out["step_executed"]
    first, second = (r["messages"] for r in server.requests)
    assert _roles(first) == ["system", "user", "assistant", "user"]
    assert _roles(second) == ["system", "user"]


def test_context_length_error_with_nothing_to_drop_is_a_request_error(tmp_path):
    game = FakeGame(tmp_path)
    err = error_response(400, '{"object":"error","param":"input_tokens","message":"too long"}')
    with FakeOpenAIServer([err]) as server:
        out = _analyze(_agent(server), game)
    assert out["retryable_failure"] and not out["step_executed"]
    assert len(server.requests) == 1


def test_request_error_reverts_history(tmp_path):
    game = FakeGame(tmp_path)
    script = [tool_call_response('print("look")'), error_response(500, "boom")]
    with FakeOpenAIServer(script) as server:
        agent = _agent(server)
        agent._state_path = game.state_path
        before = [{"role": "user", "content": "earlier"}, {"role": "assistant", "content": "ok"}]
        agent.history = list(before)
        out = _analyze(agent, game)
    assert out["retryable_failure"] and "boom" in out["error"]
    assert agent.history == before
    assert game.calls == []


def test_yield_after_turn_budget(tmp_path):
    game = FakeGame(tmp_path)
    ticks = iter([0.0, 1.0, 61.0, 61.0, 61.0])
    with FakeOpenAIServer([tool_call_response('print("x")')]) as server:
        agent = _agent(server, clock=lambda: next(ticks))
        out = _analyze(agent, game)
    assert out["yielded_control"] and out["yield_reason"] == "turn_time_budget" and not out["step_executed"]
    assert len(server.requests) == 1


def test_stop_requested_before_any_request(tmp_path):
    game = FakeGame(tmp_path)
    with FakeOpenAIServer([]) as server:
        agent = _agent(server)
        out = agent.analyze(game.state_path, 0, ["ACTION1"], game.step_env, None, 1, 5.0, lambda: True)
    assert out["yielded_control"] and server.requests == []


def test_scripted_replay_drives_multiple_turns(tmp_path):
    game = FakeGame(tmp_path)
    recording = tmp_path / "responses.jsonl"
    lines = [tool_call_response('action("UP")'), text_response("hmm"), tool_call_response('action(["DOWN", "DOWN"])')]
    recording.write_text("\n".join(json.dumps(x) for x in lines))
    with FakeOpenAIServer.from_jsonl(recording) as server:
        agent = _agent(server)
        assert _analyze(agent, game, 1)["step_executed"]
        assert _analyze(agent, game, 2)["step_executed"]
        assert _analyze(agent, game, 3)["retryable_failure"]
        assert server.exhausted
    assert [c["actions"] for c in game.calls] == [[{"action": "UP"}], [{"action": "DOWN"}, {"action": "DOWN"}]]
    assert agent.last_summary["executed_count"] == 2 and agent.last_summary["executed_actions"] == ["DOWN", "DOWN"]


def test_remaining_tool_calls_skipped_after_action(tmp_path):
    game = FakeGame(tmp_path)
    message = {"content": "", "tool_calls": [
        {"id": "a", "type": "function", "function": {"name": "python", "arguments": json.dumps({"code": 'action("UP")'})}},
        {"id": "b", "type": "function", "function": {"name": "python", "arguments": json.dumps({"code": 'action("UP")'})}},
    ]}
    with FakeOpenAIServer([chat_body(message, "tool_calls")]) as server:
        agent = _agent(server)
        _analyze(agent, game)
    assert len(game.calls) == 1
    assert [m.get("tool_call_id") for m in agent.history if m["role"] == "tool"] == ["a"]


def test_malformed_arguments_and_syntax_errors(tmp_path):
    game = FakeGame(tmp_path)
    sandbox = FakeSandbox()
    bad_json = chat_body({"content": "", "tool_calls": [
        {"id": "x", "type": "function", "function": {"name": "python", "arguments": "{not json"}}]}, "tool_calls")
    script = [bad_json, tool_call_response("def f(:\n"), tool_call_response('action("UP")')]
    with FakeOpenAIServer(script) as server:
        agent = _agent(server, sandbox=sandbox)
        _analyze(agent, game)
    tools = [json.loads(m["content"]) for m in server.requests[2]["messages"] if m["role"] == "tool"]
    assert tools[0]["error"].startswith("Empty code")
    assert tools[1]["error"].startswith("Python syntax error:")
    assert len(sandbox.calls) == 1


def test_terminal_latch_stubs_later_actions_in_the_cell(tmp_path):
    game = FakeGame(tmp_path, level_len=1)
    code = 'a = action("UP")\nb = action("UP")\nresult = b'
    with FakeOpenAIServer([tool_call_response(code)]) as server:
        agent = _agent(server)
        _analyze(agent, game)
    assert len(game.calls) == 1
    tool = json.loads(agent.history[-1]["content"])
    assert tool["result"]["executed"] is False and tool["result"]["stop_reason"] == "previous_level_completed"
    assert tool["action_calls"] == 2


def test_tool_output_truncated(tmp_path):
    game = FakeGame(tmp_path)
    with FakeOpenAIServer([tool_call_response('print("z" * 9000)\naction("UP")')]) as server:
        agent = _agent(server)
        _analyze(agent, game)
    stdout = json.loads(agent.history[-1]["content"])["stdout"]
    assert stdout.startswith("z" * 4096) and "truncated 4904" in stdout


def test_new_state_path_resets_session(tmp_path):
    game = FakeGame(tmp_path)
    with FakeOpenAIServer([tool_call_response('action("UP")')]) as server:
        agent = _agent(server)
        agent.history = [{"role": "user", "content": "stale"}]
        agent.notes = {"plan": "stale"}
        agent.generated_tokens = 99
        _analyze(agent, game)
    assert agent.history[0]["content"] != "stale" and agent.notes == {} and agent.generated_tokens == 10


# --------------------------------------------------------------------------- (b) eviction


def _msg(role, text="x", **extra):
    return {"role": role, "content": text, **extra}


def test_block_drop_rule():
    hist = [_msg("assistant"), _msg("tool"), _msg("tool"), _msg("user", "u1"), _msg("assistant")]
    assert drop_oldest_block(hist) == hist[3:]
    hist = [_msg("user", "u0"), _msg("tool"), _msg("assistant"), _msg("tool"), _msg("user", "u1"), _msg("assistant")]
    assert drop_oldest_block(hist) == hist[4:]
    only = [_msg("user", "u0"), _msg("assistant"), _msg("tool")]
    assert drop_oldest_block(only) == [only[-1]]  # never the newest message
    assert drop_oldest_block([_msg("user")]) == [_msg("user")]


def test_trim_respects_budget_and_user_first():
    system = _msg("system", "s")
    history = []
    for i in range(10):
        history += [_msg("user", f"u{i}" * 200), _msg("assistant", "a" * 600), _msg("tool", "t" * 600)]
    newest = _msg("user", "now")
    messages = [system, *history, newest]
    budget = estimate_tokens([system, *history[-6:], newest]) + 1
    trimmed = trim_messages(messages, budget)
    assert trimmed[0] is system and trimmed[-1] is newest
    assert trimmed[1]["role"] == "user"
    assert estimate_tokens(trimmed) <= budget
    assert trimmed == [system, *history[-6:], newest]
    tiny = trim_messages(messages, 1)
    assert tiny == [system, newest]


def test_image_counts_in_estimate():
    url = "data:image/png;base64," + "A" * 30000
    plain = [_msg("user", [{"type": "text", "text": "hi"}])]
    image = [_msg("user", [{"type": "text", "text": "hi"}, {"type": "image_url", "image_url": {"url": url}}])]
    assert estimate_tokens(image) - estimate_tokens(plain) >= 30000 / 3


def test_estimate_formula_includes_tools():
    messages = [_msg("user", "hi")]
    expected = len(json.dumps({"messages": messages, "tools": agent_mod.TOOLS, "tool_choice": "auto"},
                              sort_keys=True, ensure_ascii=True)) / 3
    assert estimate_tokens(messages) == expected
    assert CONFIG.budget == 31744


def test_thirty_assistant_cap_prepends_user():
    messages = [_msg("system"), _msg("user", "turn")]
    for i in range(35):
        messages += [_msg("assistant", f"a{i}"), _msg("tool", f"t{i}")]
    kept = persistable_history(messages)
    assert sum(m["role"] == "assistant" for m in kept) == 30
    assert kept[0] == _msg("user", "turn") and kept[1]["content"] == "a5"


def test_persist_drops_leading_non_user():
    kept = persistable_history([_msg("system"), _msg("tool"), _msg("assistant"), _msg("user", "u"), _msg("assistant")])
    assert _roles(kept) == ["user", "assistant"]


def test_history_is_evicted_by_images_across_turns(tmp_path):
    game = FakeGame(tmp_path)
    script = [tool_call_response('action("UP")') for _ in range(4)]
    with FakeOpenAIServer(script) as server:
        agent = _agent(server)
        probe = agent._state_path
        assert probe is None
        # A window that holds roughly two turns (each carries an image of ~1-2k chars).
        agent.config = replace(agent.config, context_window=agent.config.reply_reserve + 512 + 2400)
        for step in range(1, 5):
            assert _analyze(agent, game, step)["step_executed"]
    counts = [len(r["messages"]) for r in server.requests]
    assert counts[0] == 2 and max(counts) < 2 + 3 * 3
    for body in server.requests:
        assert body["messages"][1]["role"] == "user"
        assert estimate_tokens(body["messages"]) <= agent.config.budget or len(body["messages"]) == 2


# --------------------------------------------------------------------------- (c) carried notes


def test_extract_notes_labels_bullets_continuations():
    text = (
        "Some intro.\n- **World model:** red block moves with arrows\n  walls are grey\n\n"
        "goal model: reach the yellow cell\n* Plan: UP x3\n1. Cross-level notes: timer bar on top\nunrelated"
    )
    notes = extract_notes(text)
    assert notes["world"] == "red block moves with arrows\nwalls are grey"
    assert notes["goal"] == "reach the yellow cell"
    assert notes["plan"] == "UP x3"
    assert notes["cross_level"] == "timer bar on top\nunrelated"


def test_notes_carried_then_cleared_on_level_transition(tmp_path):
    game = FakeGame(tmp_path, level_len=2)
    text = "World model: marker moves\nPlan: go up\nCross-level notes: levels take 2 moves"
    script = [
        tool_call_response('action("UP")', content=text),
        tool_call_response('action("UP")', content="Plan: up again"),
        tool_call_response('action("DOWN")'),
    ]
    with FakeOpenAIServer(script) as server:
        agent = _agent(server)
        _analyze(agent, game, 1)
        assert agent.notes == {"world": "marker moves", "plan": "go up", "cross_level": "levels take 2 moves"}
        _analyze(agent, game, 2)
        assert agent.last_summary["level_transition"]
        assert agent.notes == {"cross_level": "levels take 2 moves"}
        _analyze(agent, game, 3)
    second_prompt = server.requests[1]["messages"][-1]["content"][0]["text"]
    third_prompt = server.requests[2]["messages"][-1]["content"][0]["text"]
    assert "World model: marker moves" in second_prompt and "Plan: go up" in second_prompt
    assert "World model:" not in third_prompt and "Cross-level notes: levels take 2 moves" in third_prompt
    assert "Level 2" in third_prompt


def test_notes_cleared_on_game_over(tmp_path):
    game = FakeGame(tmp_path, game_over_at=1)
    with FakeOpenAIServer([tool_call_response('action("UP")', content="Goal model: unknown")]) as server:
        agent = _agent(server)
        _analyze(agent, game)
    assert agent.last_summary["game_over"] and "goal" not in agent.notes


# --------------------------------------------------------------------------- gate integration


def test_system_prompt_addenda_by_mode(tmp_path):
    with FakeOpenAIServer([]) as server:
        off = _agent(server)
        forced = _agent(server, gate=WorldModelGate(probe_actions=1, max_refusals=8))
        optional = _agent(server, gate=WorldModelGate(probe_actions=1, max_refusals=8, optional=True))
    assert off.system_prompt == prompts.SYSTEM_PROMPT_BASE
    assert forced.system_prompt.endswith(prompts.GATE_ADDENDUM_FORCED)
    assert optional.system_prompt.endswith(prompts.GATE_ADDENDUM_OPTIONAL)
    assert "simulate(" in optional.system_prompt and "simulate(" not in forced.system_prompt


def test_gate_from_env(monkeypatch):
    monkeypatch.setenv("PROBE_ENFORCE", "1")
    monkeypatch.setenv("PROBE_ACTIONS", "8")
    monkeypatch.setenv("PROBE_OPTIONAL", "0")
    agent = ToolAgent(config=CONFIG, sandbox=(FakeSandbox(), FakeRefused))
    assert agent.gate is not None and agent.gate.probe_actions == 8 and agent.gate_mode == "forced"
    monkeypatch.setenv("PROBE_ENFORCE", "0")
    assert ToolAgent(config=CONFIG, sandbox=(FakeSandbox(), FakeRefused)).gate is None


PREDICT_CELL = '''
def predict(ascii, action):
    rows = ascii.splitlines()
    if action == "UP":
        return "\\n".join(rows[1:] + rows[:1])
    return "\\n".join(rows[-1:] + rows[:-1])

result = "defined"
'''


def test_gate_prepend_backtest_refusal_and_log(tmp_path):
    sandbox = FakeSandbox()
    gate = WorldModelGate(probe_actions=1, max_refusals=8)
    game = FakeGame(tmp_path, gate=gate)
    script = [
        tool_call_response('action("UP")'),
        tool_call_response(PREDICT_CELL),
        tool_call_response('action("UP")'),
        error_response(503, "stop here"),
    ]
    with FakeOpenAIServer(script) as server:
        agent = _agent(server, gate=gate, sandbox=sandbox)
        assert _analyze(agent, game, 1)["step_executed"]
        out = _analyze(agent, game, 2)
    assert out["retryable_failure"] and len(game.calls) == 1
    # every model cell is prefixed with the stored definitions (skeleton, then the captured predict)
    model_cells = [c for c in sandbox.calls if BACKTEST_DRIVER not in c["code"]]
    assert model_cells[0]["code"].startswith("def predict(ascii, action):\n    # TODO")
    assert model_cells[0]["code"].endswith('action("UP")')
    # the backtest ran in a separate sandbox call with the driver and a refusing handler
    backtests = [c for c in sandbox.calls if BACKTEST_DRIVER in c["code"]]
    assert backtests and "return ascii" not in backtests[0]["code"].split(BACKTEST_DRIVER)[0].split("def predict")[1][:80]
    with pytest.raises(FakeRefused):
        backtests[0]["handler"](["UP"])
    # the refusal reached the model as a tool error
    tools = [json.loads(m["content"]) for m in server.requests[3]["messages"] if m["role"] == "tool"]
    assert "World-model gate" in tools[-1]["error"] and "RuntimeError" in tools[-1]["error"]
    events = [json.loads(line)["event"] for line in (game.state_path.parent / "g1_p0_wm_gate.jsonl").read_text().splitlines()]
    assert events[0] == "session_start" and "captured" in events and "refused" in events


def test_agent_leaves_gate_accounting_to_step_env(tmp_path):
    game = FakeGame(tmp_path, level_len=2)
    gate = WorldModelGate(probe_actions=1, max_refusals=8)
    with FakeOpenAIServer([tool_call_response('action(["UP", "UP", "UP"])')]) as server:
        agent = _agent(server, gate=gate)
        _analyze(agent, game)
    # this step_env has no gate, so a batch over PROBE_ACTIONS ran and nothing was counted
    assert gate.level_actions == 0 and gate.level is None and gate.refusals == 0
    # the batch stopped at the level completion after 2 of 3 actions
    tool = json.loads(agent.history[-1]["content"])["action_result"]
    assert tool["requested_count"] == 3 and tool["executed_count"] == 2 and tool["stopped_early"] is True


def test_optional_gate_prepends_simulate(tmp_path):
    game = FakeGame(tmp_path)
    sandbox = FakeSandbox()
    gate = WorldModelGate(probe_actions=1, max_refusals=8, optional=True)
    with FakeOpenAIServer([tool_call_response('r = simulate(["UP"])\naction("UP")')]) as server:
        agent = _agent(server, gate=gate, sandbox=sandbox)
        assert _analyze(agent, game)["step_executed"]
    assert "def simulate(" in sandbox.calls[0]["code"] and "__WM_CONFIRMED = False" in sandbox.calls[0]["code"]
