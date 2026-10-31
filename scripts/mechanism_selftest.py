"""Plan C self-test: drive every WM_MECHANISMS=all path through scripted games and fail if any never fires.

    python scripts/mechanism_selftest.py        (from the repo root)

A detector that never fires can't be told from one that is wired wrong (lesson from the earlier harnesses),
so each path is forced at least once with the fake model server and fake games from tests/, and counted from
the logs the harness itself writes (gate log, progress log, request log, tool results, kept history).
Exit status 1 if any count is 0.
"""
from __future__ import annotations

import json
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests.fake_openai import FakeOpenAIServer, error_response, tool_call_response
from tests.plan_c_fakes import BLOCK_PREDICT, BlockGame
from tests.test_agent import FakeRefused, FakeSandbox
from wmagent.agent import BOARD_IMAGE_OMITTED, CONFIG, ToolAgent
from wmagent.world_model_gate import WorldModelGate

PATHS = ("capture_on_deny", "accepted_notice", "deny_evidence", "grant", "grant_revoked", "release_valve",
         "noop_shadow", "stuck", "continuation", "image_elision", "level_memory")


def _agent(server, gate, clock=None):
    config = replace(CONFIG, base_url=server.base_url, mechanisms="all")
    kwargs = {"clock": clock} if clock else {}
    return ToolAgent(config=config, gate=gate, sandbox=(FakeSandbox(), FakeRefused), **kwargs)


def _turn(agent, game, step):
    return agent.analyze(state_path=game.state_path, action_count=game.action_num,
                         valid_actions=["ACTION1", "ACTION2", "RESET"], step_env=game.step_env,
                         transcript_path=None, analysis_step=step, request_timeout=5.0, should_stop=lambda: False)


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []


def _tools(agent) -> list[dict]:
    return [json.loads(m["content"]) for m in agent.history if m["role"] == "tool"]


def gate_lifecycle(root: Path) -> dict[str, int]:
    """Probe 3 moves; a refused cell's predict is captured and accepted; 3 held-out hits confirm it; the level
    completes; the confirmed model carries a grant into level 2, where the mirrored board revokes it."""
    game = BlockGame(root, level_len=7)
    gate = WorldModelGate(probe_actions=3, max_refusals=8, mechanisms="all")
    script = [
        tool_call_response('action(["UP", "UP", "UP"])'),
        tool_call_response(BLOCK_PREDICT + '\naction("UP")\n'),  # refused (budget spent), predict captured
        tool_call_response('action(["DOWN", "DOWN", "DOWN"])'),  # held-out hits
        tool_call_response('action("DOWN")'),  # completes level 1
        tool_call_response('action(["UP", "UP"])'),  # level 2: grant, mirrored moves miss
        tool_call_response('result = "look"'),  # a cell after the misses: the carried model is re-judged
        error_response(503, "end"),
    ]
    counts = dict.fromkeys(("capture_on_deny", "accepted_notice", "deny_evidence", "grant", "grant_revoked",
                            "image_elision", "level_memory"), 0)
    with FakeOpenAIServer(script) as server:
        agent = _agent(server, gate)
        seen: dict[str, dict] = {}  # kept history repeats earlier tool results; count each once
        for step in range(1, 7):
            _turn(agent, game, step)
            seen.update((json.dumps(t, sort_keys=True), t) for t in _tools(agent))
            counts["image_elision"] += any(p.get("text") == BOARD_IMAGE_OMITTED for m in agent.history
                                           if m["role"] == "user" and isinstance(m["content"], list)
                                           for p in m["content"])
    tool_results = list(seen.values())
    events = [e["event"] for e in _jsonl(gate.log_path)]
    counts["capture_on_deny"] = sum(1 for i, e in enumerate(events) if e == "captured" and "refused" in events[:i])
    counts["accepted_notice"] = sum("accepted on this level" in (t.get("gate") or "") for t in tool_results)
    counts["deny_evidence"] = sum("No predict has been judged on this level yet" in (t.get("error") or "")
                                  or "Your last predict matched" in (t.get("error") or "") for t in tool_results)
    counts["grant"] = events.count("grant")
    counts["grant_revoked"] = events.count("grant_revoked")
    progress = _jsonl(game.state_path.with_name("g1_p0_progress.jsonl"))
    counts["level_memory"] = sum(e["event"] == "level_memory" for e in progress)
    return counts


def release_valve(root: Path) -> dict[str, int]:
    """One probe action, a refused action(), then 8 model calls on the blocked level release it (B is the
    profile's blocked_call_release, the refusal cap of 8; only one refusal happens, so the cap itself can't)."""
    game = BlockGame(root)
    gate = WorldModelGate(probe_actions=1, max_refusals=8, mechanisms="all")
    script = [tool_call_response('action("UP")'), tool_call_response('action("UP")')]
    script += [tool_call_response(f'result = {i}') for i in range(8)] + [error_response(503, "end")]
    with FakeOpenAIServer(script) as server:
        agent = _agent(server, gate)
        _turn(agent, game, 1)
        _turn(agent, game, 2)
    return {"release_valve": [e["event"] for e in _jsonl(gate.log_path)].count("released_blocked_calls")}


def noop_and_stuck(root: Path) -> dict[str, int]:
    """UP into the top edge changes nothing twice (STUCK), and the next UP on that board is a shadow no-op."""
    game = BlockGame(root)
    script = [tool_call_response('action(["UP"] * 6)'), tool_call_response('action("UP")'),
              error_response(503, "end")]
    with FakeOpenAIServer(script) as server:
        agent = _agent(server, None)
        _turn(agent, game, 1)
        _turn(agent, game, 2)
    progress = [e["event"] for e in _jsonl(game.state_path.with_name("g1_p0_progress.jsonl"))]
    return {"noop_shadow": min(progress.count("noop_would_deny"), progress.count("noop_outcome")),
            "stuck": progress.count("stuck")}


def continuation(root: Path) -> dict[str, int]:
    """A turn that inspects and then hits the 60 s budget is continued with the short prompt."""
    game = BlockGame(root)
    ticks = iter([0.0, 1.0, 61.0, 61.0, 61.0] + [100.0] * 20)
    script = [tool_call_response('print("look")'), tool_call_response('action("UP")')]
    with FakeOpenAIServer(script) as server:
        agent = _agent(server, None, clock=lambda: next(ticks))
        _turn(agent, game, 1)
        _turn(agent, game, 1)
        user = [m for m in server.requests[1]["messages"] if m["role"] == "user"][-1]
    return {"continuation": int("Your turn is still open" in user["content"][0]["text"])}


def run() -> dict[str, int]:
    counts: dict[str, int] = {}
    with tempfile.TemporaryDirectory() as tmp:
        for scenario in (gate_lifecycle, release_valve, noop_and_stuck, continuation):
            folder = Path(tmp) / scenario.__name__
            folder.mkdir()
            counts.update(scenario(folder))
    return {path: counts.get(path, 0) for path in PATHS}


def main() -> int:
    counts = run()
    for path, count in counts.items():
        print(f"{path:16} {count}")
    missing = [p for p, c in counts.items() if c == 0]
    print("FAIL: never fired: " + ", ".join(missing) if missing else "OK: every path fired")
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
