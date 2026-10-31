"""A chat request that fails still gets its row in <stem>_requests.jsonl, with the error; the turn is still
reverted and reported as before. In arms A-C the last call of almost every game (cut at the 2,700 s cap) left a
snapshot but no request row."""
import json
from dataclasses import replace

from tests.fake_openai import FakeOpenAIServer, error_response, tool_call_response
from tests.test_agent import FakeGame, FakeRefused, FakeSandbox, _analyze
from wmagent.agent import CONFIG, ToolAgent


def test_a_failed_request_is_logged_with_its_error_and_the_turn_still_reverts(tmp_path):
    game = FakeGame(tmp_path)
    script = [tool_call_response('print("look")'), error_response(503, "server busy")]
    with FakeOpenAIServer(script) as server:
        agent = ToolAgent(config=replace(CONFIG, base_url=server.base_url), gate=None,
                          sandbox=(FakeSandbox(), FakeRefused))
        outcome = _analyze(agent, game, 1)
    assert outcome["retryable_failure"] and agent.history == []
    rows = [json.loads(x) for x in (game.state_path.parent / "g1_p0_requests.jsonl").read_text().splitlines()]
    assert [r["request"] for r in rows] == [1, 2] and "error" not in rows[0]
    assert rows[1]["error"].startswith("AnalyzerRequestError: HTTP 503") and rows[1]["completion_tokens"] is None
    assert rows[1]["turn_request"] == 2 and rows[1]["t_end"] >= rows[1]["t_start"]
