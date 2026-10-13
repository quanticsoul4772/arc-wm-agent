"""Tests for wmagent.session (spec 2.3, acceptance T2 a-e). The engine and agent are fakes."""
from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from arcengine import GameAction, GameState

from wmagent.session import (
    ACTION_PAYLOAD_KEYS,
    BATCH_PAYLOAD_KEYS,
    ERROR_PAYLOAD_KEYS,
    TERMINAL_PAYLOAD_KEYS,
    GameSession,
    SessionCrashed,
    make_play,
)

SIMPLE = [1, 2, 3, 4, 5, 6]  # ACTION7 is deliberately not available


@dataclass
class Obs:
    frame: list
    state: GameState
    levels_completed: int
    available_actions: list[int]


@dataclass
class FakeGame:
    """Scripted engine: `effects` maps an engine action name to what it does when executed.

    Effects: "move" (board changes), "noop", "level" (+1 level), "over" (GAME_OVER), "win", "raise".
    RESET clears GAME_OVER unless `sticky_over` is set.
    """

    number_of_levels: int = 3
    effects: dict[str, str] = field(default_factory=dict)
    sticky_over: bool = False
    game_id: str = "fake-0001"
    state: str = "playing"
    levels_completed: int = 0
    solver_note: str = ""
    calls: list[tuple[str, Any, int]] = field(default_factory=list)
    finished_with: list[int] = field(default_factory=list)
    counter: int = 0

    def __post_init__(self) -> None:
        self.frame = Obs([self._grid(0)], GameState.NOT_FINISHED, 0, self._available())

    def _available(self) -> list[int]:
        # effect "unavailable" removes an action; "available" adds one (e.g. ACTION7)
        extra = [GameAction[name].value for name, effect in self.effects.items() if effect == "available"]
        ids = [a for a in SIMPLE if self.effects.get(f"ACTION{a}") != "unavailable"] + extra
        return ids

    @staticmethod
    def _grid(mark: int) -> list[list[int]]:
        grid = [[0] * 64 for _ in range(64)]
        grid[mark % 64][0] = 5
        return grid

    def execute(self, action: GameAction, data: dict | None = None, *, generated_tokens: int = 0) -> Obs:
        if action != GameAction.RESET and action.value not in self.frame.available_actions:
            raise ValueError("not available")
        effect = "reset" if action == GameAction.RESET else self.effects.get(action.name, "move")
        if effect == "raise":
            raise RuntimeError("engine exploded")
        self.calls.append((action.name, data, generated_tokens))
        state = GameState.NOT_FINISHED
        if effect != "noop":
            self.counter += 1
        if effect == "level":
            self.levels_completed += 1
        elif effect == "over":
            state = GameState.GAME_OVER
        elif effect == "win":
            self.levels_completed = self.number_of_levels
            state = GameState.WIN
            self.state = "won"
        elif effect == "reset" and self.sticky_over and self.frame.state == GameState.GAME_OVER:
            state = GameState.GAME_OVER
        self.frame = Obs([self._grid(self.counter)], state, self.levels_completed, self._available())
        return self.frame

    def mark_crashed(self, note: str) -> None:
        if self.state == "playing":
            self.state = "crashed"
            self.solver_note = note

    def finish(self, final_generated_tokens: int) -> None:
        self.finished_with.append(final_generated_tokens)
        if self.state == "playing":
            self.state = "gave_up"


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


class StubAgent:
    """Each turn runs `script(turn_index, call)`; `call` is the analyze kwargs."""

    def __init__(self, script: Callable[[int, dict], dict | None]) -> None:
        self.script = script
        self.generated_tokens = 0
        self.calls: list[dict] = []

    def analyze(self, **kwargs: Any) -> dict | None:
        self.calls.append(kwargs)
        return self.script(len(self.calls) - 1, kwargs)


def _session(tmp_path: Path, game: FakeGame | None = None, agent: StubAgent | None = None, **kw: Any) -> GameSession:
    session = GameSession(
        game=game or FakeGame(),
        agent=agent or StubAgent(lambda i, c: {"step_executed": False}),
        state_path=tmp_path / "artifacts" / "fake-0001_p0_tool_runtime_state.json",
        transcript_path=tmp_path / "transcripts" / "fake-0001_p0.txt",
        events_path=tmp_path / "artifacts" / "fake-0001_p0_events.jsonl",
        sleep=lambda s: None,
        **kw,
    )
    session.play_started = session.clock()
    session._seed()
    return session


def _turn(*actions: str) -> Callable[[int, dict], dict]:
    def script(i: int, call: dict) -> dict:
        call["step_env"]({"actions": [{"action": a} for a in actions]})
        return {"step_executed": True}

    return script


# --- (a) null-coordinate guard -----------------------------------------------------------


@pytest.mark.parametrize(
    "arguments",
    [
        {"action": "MOUSE"},
        {"action": "MOUSE", "row": None, "col": 5},
        {"action": "MOUSE", "row": 5, "col": None},
        {"action": "MOUSE", "row": None, "col": None},
        {"action": "MOUSE", "row": 5},
        {"action": "MOUSE", "col": 5},
        {"action": "MOUSE", "row": "five", "col": 5},
        {"action": "MOUSE", "row": "5.5", "col": 5},
        {"action": "MOUSE", "row": [5], "col": 5},
        {"action": "ACTION6"},
        {"action": "ACTION6", "row": None, "col": 2},
        {"action": "MOUSE", "x": 5, "y": 5},
        {"actions": [{"action": "UP"}, {"action": "MOUSE", "row": None, "col": 1}]},
    ],
)
def test_mouse_without_integer_coordinates_is_rejected_and_nothing_runs(tmp_path, arguments):
    session = _session(tmp_path)
    result = session.step_env(arguments)
    assert result["executed"] is False
    assert "row" in result["error"] and "col" in result["error"]
    assert set(result) == ERROR_PAYLOAD_KEYS
    assert session.game.calls == []


def test_mouse_coordinates_are_clamped_and_sent_as_x_col_y_row(tmp_path):
    session = _session(tmp_path)
    result = session.step_env({"action": "MOUSE", "row": -4, "col": 99})
    assert result["executed"] is True
    assert session.game.calls == [("ACTION6", {"x": 63, "y": 0}, 0)]
    assert result["action_display"] == "MOUSE(row=0, col=63)"
    assert result["action_data"] == {"x": 63, "y": 0}


@pytest.mark.parametrize(
    "row,col,expected",
    [
        (5.9, 2, {"x": 2, "y": 5}),
        ("7", "3", {"x": 3, "y": 7}),
        (True, False, {"x": 0, "y": 1}),
        (" 4 ", 70.2, {"x": 63, "y": 4}),
    ],
)
def test_mouse_coordinates_are_converted_with_int(tmp_path, row, col, expected):
    """F7: the original converts with int(), so floats truncate and numeric strings and bools pass."""
    session = _session(tmp_path)
    result = session.step_env({"action": "MOUSE", "row": row, "col": col})
    assert result["executed"] is True
    assert session.game.calls == [("ACTION6", expected, 0)]


def test_action7_is_listed_but_never_executes(tmp_path):
    """F2: only ACTION1-6 and RESET map to engine actions; ACTION7 is an unknown action."""
    session = _session(tmp_path, FakeGame(effects={"ACTION7": "available"}))
    assert "ACTION7" in session._valid_labels()
    single = session.step_env({"action": "ACTION7"})
    assert single["executed"] is False and "ACTION7" in single["error"] and set(single) == ERROR_PAYLOAD_KEYS
    batch = session.step_env({"actions": [{"action": "UP"}, {"action": "action7"}]})
    assert batch["executed"] is False and "ACTION7" in batch["error"].upper()
    assert session.game.calls == [] and session.action_count == 0
    assert "ACTION7" in single["valid_actions"]


def test_simple_action_ignores_coordinates_and_engine_names_are_accepted(tmp_path):
    session = _session(tmp_path)
    result = session.step_env({"actions": [{"action": "up", "row": None}, {"action": "ACTION2"}]})
    assert [c[:2] for c in session.game.calls] == [("ACTION1", None), ("ACTION2", None)]
    assert result["executed_actions"] == ["UP", "DOWN"]


@pytest.mark.parametrize(
    "arguments",
    [
        {},
        {"action": "UP", "actions": [{"action": "UP"}]},
        {"actions": []},
        {"actions": "UP"},
        {"action": "JUMP"},
        {"action": None},
        {"actions": [{"action": "UP"}, {"action": "FLY"}]},
    ],
)
def test_malformed_arguments_give_error_payload(tmp_path, arguments):
    session = _session(tmp_path)
    result = session.step_env(arguments)
    assert set(result) == ERROR_PAYLOAD_KEYS
    assert result["executed"] is False
    assert session.game.calls == []


# --- (b) batch semantics -----------------------------------------------------------------

BATCH_TABLE = [
    # id, effects, batch, expected engine calls, outcome
    ("invalid-first", {"ACTION5": "unavailable"}, ["SPACE", "UP"], [], {"error": True}),
    ("unknown-action7", {"ACTION7": "available"}, ["UP", "ACTION7"], [], {"error": True}),
    ("engine-error-first", {"ACTION1": "raise"}, ["UP", "DOWN"], [], {"error": True}),
    (
        "invalid-later",
        {"ACTION5": "unavailable"},
        ["UP", "SPACE", "DOWN"],
        ["ACTION1"],
        {"stop_reason": "invalid_action", "executed_count": 1, "stopped_early": True},
    ),
    (
        "engine-error-later",
        {"ACTION2": "raise"},
        ["UP", "DOWN", "LEFT"],
        ["ACTION1"],
        {"stop_reason": "action_error", "executed_count": 1, "stopped_early": True},
    ),
    (
        "level-complete-mid-batch",
        {"ACTION1": "level"},
        ["UP", "DOWN"],
        ["ACTION1"],
        {"stop_reason": "level_completed", "level_completed": True, "level": 2, "score": 1, "executed_count": 1},
    ),
    (
        "game-over-mid-batch",
        {"ACTION3": "over"},
        ["UP", "LEFT", "DOWN"],
        ["ACTION1", "ACTION3"],
        {"stop_reason": "game_over", "game_over": True, "done": False, "state": "GAME_OVER", "executed_count": 2},
    ),
    (
        "win",
        {"ACTION4": "win"},
        ["UP", "RIGHT", "DOWN"],
        ["ACTION1", "ACTION4"],
        {"stop_reason": "run_complete", "run_complete": True, "done": True, "level_completed": False, "state": "WIN"},
    ),
    (
        "full-batch",
        {"ACTION2": "noop"},
        ["UP", "DOWN", "LEFT"],
        ["ACTION1", "ACTION2", "ACTION3"],
        {"stop_reason": None, "stopped_early": False, "executed_count": 3, "board_changed": True},
    ),
]


@pytest.mark.parametrize("effects,batch,expected_calls,outcome", [r[1:] for r in BATCH_TABLE], ids=[r[0] for r in BATCH_TABLE])
def test_batch_semantics(tmp_path, effects, batch, expected_calls, outcome):
    session = _session(tmp_path, FakeGame(effects=effects))
    result = session.step_env({"actions": [{"action": a} for a in batch]})
    assert [c[0] for c in session.game.calls] == expected_calls
    if outcome.get("error"):
        assert result["executed"] is False and set(result) == ERROR_PAYLOAD_KEYS
        return
    assert result["executed"] is True
    assert result["requested_count"] == len(batch)
    assert result["executed_actions"] == [r for r in result["requested_actions"]][: len(expected_calls)]
    for key, value in outcome.items():
        assert result[key] == value, key
    assert session.action_count == len(expected_calls)
    assert len(session.history) == 1 + len(expected_calls)


def test_batch_reward_is_summed_and_last_reward_kept(tmp_path):
    session = _session(tmp_path, FakeGame(number_of_levels=4))
    session.step_env({"action": "UP"})
    session.game.effects = {"ACTION2": "level"}
    result = session.step_env({"actions": [{"action": "UP"}, {"action": "DOWN"}]})
    assert result["reward"] == 0.25 and result["last_reward"] == 0.25
    assert result["batched"] is True


def test_already_game_over_or_stopping_returns_terminal_payload(tmp_path):
    session = _session(tmp_path, FakeGame(effects={"ACTION1": "over"}))
    session.step_env({"action": "UP"})
    over = session.step_env({"action": "DOWN"})
    assert set(over) == TERMINAL_PAYLOAD_KEYS
    assert over["executed"] is False and over["stop_reason"] == "game_over"

    stopped = _session(tmp_path / "s", stop_flag=threading.Event())
    stopped.stop_flag.set()
    result = stopped.step_env({"action": "UP"})
    assert result["stop_reason"] == "stopped" and stopped.game.calls == []

    won = _session(tmp_path / "w", FakeGame(effects={"ACTION1": "win"}))
    won.step_env({"action": "UP"})
    assert won.step_env({"action": "UP"})["stop_reason"] == "run_complete"


def test_token_attribution_first_action_of_batch_carries_turn_tokens(tmp_path):
    session = _session(tmp_path)
    session.agent.generated_tokens = 120
    session.step_env({"actions": [{"action": "UP"}, {"action": "DOWN"}]})
    session.agent.generated_tokens = 150
    session.step_env({"action": "LEFT"})
    assert [c[2] for c in session.game.calls] == [120, 0, 30]


def test_runtime_state_and_events_are_written_per_action(tmp_path):
    session = _session(tmp_path)
    session.step_env({"action": "MOUSE", "row": 3, "col": 7})
    state = json.loads(session.state_path.read_text())
    assert set(state) == {"current_frame", "history"}
    assert [h["action"] for h in state["history"]] == ["", "MOUSE(row=3, col=7)"]
    assert state["current_frame"] == state["history"][-1]["frame"]
    assert set(state["current_frame"]) == {"grid", "step", "level"}
    assert state["current_frame"]["step"] == 1 and state["current_frame"]["level"] == 1
    events = [json.loads(line) for line in session.events_path.read_text().splitlines()]
    assert [e["type"] for e in events] == ["initial", "action"]
    assert events[1]["action_display"] == "MOUSE(row=3, col=7)"
    assert len(events[1]["board_ascii"].splitlines()) == 64


# --- (d) payload key sets ----------------------------------------------------------------


def test_payload_key_sets_match_spec(tmp_path):
    assert ACTION_PAYLOAD_KEYS == {
        "executed", "action_num", "level", "score", "reward", "state", "valid_actions",
        "board_changed", "done", "level_completed", "game_over", "run_complete",
        "action_name", "action_data", "action_display", "batch_index", "batch_size",
        "run_elapsed_seconds", "time_remaining_seconds",
    }  # fmt: skip
    assert BATCH_PAYLOAD_KEYS - ACTION_PAYLOAD_KEYS == {
        "last_reward", "batched", "requested_count", "executed_count",
        "requested_actions", "executed_actions", "stopped_early", "stop_reason",
    }  # fmt: skip
    session = _session(tmp_path)
    assert set(session.step_env({"action": "UP"})) == BATCH_PAYLOAD_KEYS
    assert set(session.step_env({"actions": [{"action": "UP"}]})) == BATCH_PAYLOAD_KEYS
    assert set(session.step_env({"action": "ACTION7"})) == ERROR_PAYLOAD_KEYS
    assert session.step_env({"action": "UP"})["valid_actions"] == ["UP", "DOWN", "LEFT", "RIGHT", "SPACE", "MOUSE"]
    no_space = _session(tmp_path / "n", FakeGame(effects={"ACTION5": "unavailable"}))
    early = no_space.step_env({"actions": [{"action": "UP"}, {"action": "SPACE"}]})
    assert set(early) == BATCH_PAYLOAD_KEYS | {"stop_detail"}


def test_done_is_true_only_on_win(tmp_path):
    """F6: GAME_OVER is recoverable, so `done` stays False there (also in the terminal payload)."""
    session = _session(tmp_path, FakeGame(effects={"ACTION1": "over"}))
    over = session.step_env({"action": "UP"})
    assert over["game_over"] is True and over["done"] is False
    assert session.step_env({"action": "DOWN"})["done"] is False  # terminal payload while GAME_OVER
    won = _session(tmp_path / "w", FakeGame(effects={"ACTION4": "win"})).step_env({"action": "RIGHT"})
    assert won["run_complete"] is True and won["done"] is True


# --- (c) auto-reset ----------------------------------------------------------------------


def test_auto_reset_fires_once_per_game_over_without_consulting_agent(tmp_path):
    game = FakeGame(effects={"ACTION3": "over"})
    flag = threading.Event()

    def script(i: int, call: dict) -> dict:
        if i in (0, 2):
            call["step_env"]({"actions": [{"action": "UP"}, {"action": "LEFT"}]})
        elif i == 1:
            call["step_env"]({"action": "DOWN"})
        else:
            flag.set()
        return {"step_executed": True}

    agent = StubAgent(script)
    GameSession(
        game=game,
        agent=agent,
        state_path=tmp_path / "s.json",
        transcript_path=tmp_path / "t.txt",
        events_path=tmp_path / "e.jsonl",
        stop_flag=flag,
    ).run()
    names = [c[0] for c in game.calls]
    assert names == ["ACTION1", "ACTION3", "RESET", "ACTION2", "ACTION1", "ACTION3", "RESET"]
    assert all(c[2] == 0 for c in game.calls if c[0] == "RESET")
    assert len(agent.calls) == 4


def test_auto_reset_is_not_repeated_when_reset_leaves_game_over(tmp_path):
    game = FakeGame(effects={"ACTION1": "over"}, sticky_over=True)
    flag = threading.Event()

    def script(i: int, call: dict) -> dict:
        if i == 0:
            call["step_env"]({"action": "UP"})
        elif i >= 2:
            flag.set()
        return {"step_executed": True}

    GameSession(
        game=game,
        agent=StubAgent(script),
        state_path=tmp_path / "s.json",
        transcript_path=tmp_path / "t.txt",
        events_path=tmp_path / "e.jsonl",
        stop_flag=flag,
    ).run()
    assert [c[0] for c in game.calls] == ["ACTION1", "RESET"]


# --- (e) time cap and turn-loop results ---------------------------------------------------


def test_time_cap_stops_turn_loop(tmp_path):
    clock = FakeClock()

    def script(i: int, call: dict) -> dict:
        assert call["request_timeout"] == pytest.approx(min(900.0, max(0.1, 250.0 - 100.0 * i)))
        clock.now += 100.0
        call["step_env"]({"action": "UP"})
        return {"step_executed": True}

    agent = StubAgent(script)
    game = FakeGame()
    session = GameSession(
        game=game,
        agent=agent,
        state_path=tmp_path / "a" / "s.json",
        transcript_path=tmp_path / "t.txt",
        events_path=tmp_path / "e.jsonl",
        time_cap_s=250.0,
        clock=clock,
    )
    session.run()
    assert len(agent.calls) == 3
    assert [c["analysis_step"] for c in agent.calls] == [1, 2, 3]
    # The third turn's action ran past the cap: the step_env stop check refused it.
    assert len(game.calls) == 2
    assert game.state == "gave_up" and game.finished_with == [0]
    assert game.solver_note == "tokens=0"
    assert not session.state_path.exists()


def test_action_cap_stops_turn_loop(tmp_path):
    game = FakeGame()
    agent = StubAgent(_turn("UP", "DOWN"))
    GameSession(
        game=game, agent=agent, state_path=tmp_path / "s.json", transcript_path=tmp_path / "t.txt",
        events_path=tmp_path / "e.jsonl", max_actions=5,
    ).run()  # fmt: skip
    assert len(game.calls) == 5 and len(agent.calls) == 3


def test_retry_and_yield_keep_the_analysis_step(tmp_path):
    sleeps: list[float] = []
    results = [{"retryable_failure": True}, {"yielded_control": True}, {"step_executed": False}, None]
    agent = StubAgent(lambda i, c: results[i])
    agent.generated_tokens = 0
    game = FakeGame()
    session = GameSession(
        game=game, agent=agent, state_path=tmp_path / "s.json", transcript_path=tmp_path / "t.txt",
        events_path=tmp_path / "e.jsonl", sleep=sleeps.append,
    )  # fmt: skip
    with pytest.raises(SessionCrashed):
        session.run()
    assert [c["analysis_step"] for c in agent.calls] == [1, 1, 1, 2]
    assert sleeps == [1.0]
    assert game.state == "crashed" and game.solver_note.startswith("error: SessionCrashed")
    assert game.finished_with == [0]


def test_win_ends_loop_and_final_tokens_go_to_finish(tmp_path):
    game = FakeGame(effects={"ACTION4": "win"})

    def script(i: int, call: dict) -> dict:
        agent.generated_tokens += 40
        assert call["valid_actions"] == ["ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5", "ACTION6"]
        call["step_env"]({"action": "RIGHT" if i == 1 else "UP"})
        agent.generated_tokens += 7
        return {"step_executed": True}

    agent = StubAgent(script)
    GameSession(
        game=game, agent=agent, state_path=tmp_path / "s.json", transcript_path=tmp_path / "t.txt",
        events_path=tmp_path / "e.jsonl",
    ).run()  # fmt: skip
    assert [c[2] for c in game.calls] == [40, 47]
    assert game.state == "won" and game.finished_with == [7]
    assert game.solver_note == "tokens=94"


# --- the gate is not consulted by step_env ------------------------------------------------


def test_step_env_does_no_gate_accounting(tmp_path):
    """The agent's action() handler owns the gate (F1/F3); step_env runs batches of any size."""
    game = FakeGame(effects={"ACTION2": "level"})
    session = _session(tmp_path, game)
    assert not hasattr(session, "gate")
    result = session.step_env({"actions": [{"action": "UP"}] * 4 + [{"action": "DOWN"}]})
    assert result["executed_count"] == 5 and result["level_completed"] is True


def test_make_play_builds_session_from_runner_context(tmp_path):
    clock_start = time.monotonic()
    game = FakeGame()
    agents: list[StubAgent] = []

    def factory() -> StubAgent:
        agents.append(StubAgent(_turn("UP")))
        return agents[-1]

    ctx = SimpleNamespace(
        game=game,
        state_path=tmp_path / "artifacts" / "g_p0_tool_runtime_state.json",
        transcript_path=tmp_path / "transcripts" / "g_p0.txt",
        events_path=tmp_path / "artifacts" / "g_p0_events.jsonl",
        stop_event=threading.Event(),
        per_game_cap_s=None,
        soft_deadline=None,
        max_actions=3,
        analyzer_timeout_s=900.0,
        play_started=clock_start,
    )
    make_play(factory)(ctx)
    assert len(agents) == 1 and len(game.calls) == 3
    assert game.state == "gave_up" and game.finished_with == [0]
    assert not ctx.state_path.exists() and ctx.events_path.exists()


def test_time_cap_counts_from_context_play_start(tmp_path):
    ctx = SimpleNamespace(
        game=FakeGame(),
        state_path=tmp_path / "s.json",
        transcript_path=tmp_path / "t.txt",
        events_path=tmp_path / "e.jsonl",
        stop_event=threading.Event(),
        per_game_cap_s=10.0,
        soft_deadline=None,
        max_actions=None,
        analyzer_timeout_s=900.0,
        play_started=time.monotonic() - 60.0,
    )
    agent = StubAgent(_turn("UP"))
    GameSession.from_context(ctx, agent).run()
    assert agent.calls == [] and ctx.game.calls == [] and ctx.game.state == "gave_up"
