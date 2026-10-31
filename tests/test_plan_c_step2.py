"""Plan C step 2: the per-request context monitor (M4a, every arm) and the yield continuation prompt (M2c,
WM_MECHANISMS=all only)."""
import json
from dataclasses import replace

from tests.fake_openai import FakeOpenAIServer, chat_body, tool_call_response
from tests.test_agent import FakeGame, FakeRefused, FakeSandbox, _analyze
from wmagent import prompts
from wmagent.agent import CONFIG, ToolAgent
from wmagent.kaggle import setup_vllm


def _agent(server, mechanisms="base", clock=None, prompts_dir=None):
    config = replace(CONFIG, base_url=server.base_url, mechanisms=mechanisms)
    kwargs = {"clock": clock} if clock else {}
    return ToolAgent(config=config, gate=None, sandbox=(FakeSandbox(), FakeRefused), prompts_dir=prompts_dir,
                     **kwargs)


def _user_texts(request):
    out = []
    for m in request["messages"]:
        if m["role"] == "user":
            content = m["content"]
            out.append(content if isinstance(content, str) else content[0]["text"])
    return out


def _yield_then_play(tmp_path, mechanisms, prompts_dir=None):
    """Turn 1 inspects, then the 60 s budget yields it; turn 2 (same analysis step) plays."""
    game = FakeGame(tmp_path)
    ticks = iter([0.0, 1.0, 61.0, 61.0, 61.0] + [100.0] * 20)
    script = [tool_call_response('print("look")'), tool_call_response('action("UP")')]
    with FakeOpenAIServer(script) as server:
        agent = _agent(server, mechanisms, clock=lambda: next(ticks), prompts_dir=prompts_dir)
        first = _analyze(agent, game, 1)
        second = _analyze(agent, game, 1)
        requests = server.requests
    return first, second, requests, game


def test_all_arm_continues_a_yielded_turn_with_a_short_prompt(tmp_path):
    first, second, requests, _ = _yield_then_play(tmp_path, "all")
    assert first["yielded_control"] and second["step_executed"]
    last_user = _user_texts(requests[1])[-1]
    assert last_user.startswith("Step 1, level 1. Your turn is still open")
    assert len(last_user) < len(_user_texts(requests[0])[-1]) / 3
    # the live board image still rides along with the continuation
    user_messages = [m for m in requests[1]["messages"] if m["role"] == "user"]
    assert any(p.get("type") == "image_url" for p in user_messages[-1]["content"])


def test_base_arm_resends_the_full_prompt_after_a_yield(tmp_path):
    _, _, requests, _ = _yield_then_play(tmp_path, "base")
    assert "Your turn is still open" not in _user_texts(requests[1])[-1]
    assert _user_texts(requests[1])[-1].startswith("Step 1, level 1.")


def test_no_continuation_after_a_reverted_history(tmp_path):
    game = FakeGame(tmp_path)
    two_calls = {"content": "", "tool_calls": [
        {"id": "a", "type": "function", "function": {"name": "python", "arguments": json.dumps({"code": "x = 1"})}},
        {"id": "b", "type": "function", "function": {"name": "python", "arguments": json.dumps({"code": "y = 2"})}},
    ]}
    ticks = iter([0.0, 1.0, 61.0, 61.0, 61.0, 61.0] + [100.0] * 20)
    script = [chat_body(two_calls, "tool_calls"), tool_call_response('action("UP")')]
    with FakeOpenAIServer(script) as server:
        agent = _agent(server, "all", clock=lambda: next(ticks))
        assert _analyze(agent, game, 1)["yielded_control"] and agent.history == []
        _analyze(agent, game, 1)
        assert "Your turn is still open" not in _user_texts(server.requests[1])[-1]


def test_continuation_prompt_lists_labels_and_mouse_arguments():
    text = prompts.continuation_prompt(step=4, level=2, valid_actions=["UP", "MOUSE"])
    assert text.startswith("Step 4, level 2.") and "UP, MOUSE" in text and "row and col" in text


def test_request_log_has_one_row_per_request_with_usage(tmp_path):
    _, _, _, game = _yield_then_play(tmp_path, "base")
    rows = [json.loads(x) for x in (game.state_path.parent / "g1_p0_requests.jsonl").read_text().splitlines()]
    assert [r["request"] for r in rows] == [1, 2] and [r["step"] for r in rows] == [1, 1]
    for r in rows:
        assert r["prompt_tokens"] == 100 and r["completion_tokens"] == 10 and r["cached_tokens"] is None
        assert r["t_end"] >= r["t_start"] and r["message_count"] >= 2 and r["trimmed"] == 0
        assert r["estimated_tokens"] > 0 and r["finish_reason"] == "tool_calls"


def test_snapshots_of_one_step_are_not_overwritten(tmp_path):
    _yield_then_play(tmp_path, "base", prompts_dir=tmp_path / "prompts")
    files = sorted(p.name for p in (tmp_path / "prompts" / "g1_p0").iterdir())
    assert files == ["step_0001_r00001.json", "step_0001_r00002.json"]


def test_vllm_reports_prompt_token_details(tmp_path):
    assert "--enable-prompt-tokens-details" in setup_vllm.vllm_command(tmp_path / "m")
