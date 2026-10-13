"""Game session: the turn loop around one started game, and the `step_env` callback.

Spec: track2-rewrite-spec-2026-10-13.md, section 2.3 (behaviour) and 2.7 (equivalence checklist).

The session sits between three other modules:
- the started game from the engine adapter (T1), used through the `SessionGame` protocol below;
- the agent (T4), used through the `SessionAgent` protocol below;
- the world-model gate (ours, `world_model_gate.py`) is NOT consulted here: the agent's `action()` handler
  does all of the gate's action accounting (observe_level, check_action, record_actions), as in the original.
"""

from __future__ import annotations

import json
import os
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from arcengine import GameAction, GameState

from wmagent.world_model_gate import WorldModelGate

GRID_MAX = 63

# Model-facing labels and the engine actions they stand for. Engine names are accepted as well.
LABEL_TO_ENGINE: dict[str, GameAction] = {
    "UP": GameAction.ACTION1,
    "DOWN": GameAction.ACTION2,
    "LEFT": GameAction.ACTION3,
    "RIGHT": GameAction.ACTION4,
    "SPACE": GameAction.ACTION5,
    "MOUSE": GameAction.ACTION6,
    "RESET": GameAction.RESET,
}
ENGINE_TO_LABEL: dict[GameAction, str] = {engine: label for label, engine in LABEL_TO_ENGINE.items()}
# Engine names step_env accepts as-is. ACTION7 (and anything else) is not mapped, as in the original:
# it stays listed in valid_actions but asking for it is an unknown-action error and nothing runs.
ENGINE_NAMES: dict[str, GameAction] = {action.name: action for action in LABEL_TO_ENGINE.values()}

# Index -> character for the ascii board in the events sidecar (same alphabet as the sandbox views).
ASCII_ALPHABET = "WwgGcBMPRbSYOrNp"

ACTION_PAYLOAD_KEYS = frozenset(
    {
        "executed",
        "action_num",
        "level",
        "score",
        "reward",
        "state",
        "valid_actions",
        "board_changed",
        "done",
        "level_completed",
        "game_over",
        "run_complete",
        "action_name",
        "action_data",
        "action_display",
        "batch_index",
        "batch_size",
        "run_elapsed_seconds",
        "time_remaining_seconds",
    }
)
BATCH_EXTRA_KEYS = frozenset(
    {
        "last_reward",
        "batched",
        "requested_count",
        "executed_count",
        "requested_actions",
        "executed_actions",
        "stopped_early",
        "stop_reason",
    }
)
BATCH_PAYLOAD_KEYS = ACTION_PAYLOAD_KEYS | BATCH_EXTRA_KEYS
ERROR_PAYLOAD_KEYS = frozenset({"executed", "error", "valid_actions", "run_elapsed_seconds", "time_remaining_seconds"})
TERMINAL_PAYLOAD_KEYS = frozenset(
    {
        "executed",
        "action_num",
        "level",
        "score",
        "state",
        "valid_actions",
        "done",
        "game_over",
        "run_complete",
        "stop_reason",
        "stop_detail",
        "run_elapsed_seconds",
        "time_remaining_seconds",
    }
)


class ActionRefused(Exception):
    """The world-model gate refused a batch; the sandbox surfaces it to model code as RuntimeError(msg)."""


class SessionCrashed(RuntimeError):
    """The agent returned no result for a turn."""


class SessionGame(Protocol):
    """What the session needs from a started game (`wmagent.engine.Game`, spec 2.1)."""

    game_id: str
    number_of_levels: int
    state: str  # "playing", "won", "gave_up", "cancelled" or "crashed"
    levels_completed: int
    solver_note: str
    frame: Any  # the latest FrameDataRaw: .frame (layers), .state, .levels_completed, .available_actions

    def execute(self, action: GameAction, data: dict[str, Any] | None = None, *, generated_tokens: int = 0) -> Any:
        """Run one engine action and update `frame`; raise on rejection or engine error."""

    def mark_crashed(self, note: str) -> None:
        """Set state `crashed` with `note` if still playing."""

    def finish(self, final_generated_tokens: int = 0) -> None:
        """Idempotent finish (spec 2.1)."""


class PlayContextLike(Protocol):
    """The runner's per-game context (`wmagent.runner.PlayContext`, spec 2.2)."""

    game: Any
    state_path: Path
    transcript_path: Path
    events_path: Path
    stop_event: threading.Event
    per_game_cap_s: float | None
    soft_deadline: float | None
    max_actions: int | None
    analyzer_timeout_s: float
    play_started: float


class SessionAgent(Protocol):
    """What the session needs from the agent (spec 2.4)."""

    generated_tokens: int  # cumulative generated tokens for the current game

    def analyze(
        self,
        *,
        state_path: Path,
        action_count: int,
        valid_actions: list[str],
        step_env: Callable[[Mapping[str, Any]], dict[str, Any]],
        transcript_path: Path,
        analysis_step: int,
        request_timeout: float,
        should_stop: Callable[[], bool],
    ) -> Mapping[str, Any] | None:
        """One turn. Returns {step_executed, retryable_failure, yielded_control, reasoning} or None."""


class StopFlag(Protocol):
    def is_set(self) -> bool: ...


@dataclass(frozen=True)
class _Request:
    action: GameAction
    data: dict[str, int] | None
    display: str


@dataclass
class GameSession:
    game: SessionGame
    agent: SessionAgent
    state_path: Path
    transcript_path: Path
    events_path: Path
    stop_flag: StopFlag = field(default_factory=threading.Event)
    time_cap_s: float | None = None
    max_actions: int | None = None
    analyzer_timeout_s: float = 900.0
    deadline_monotonic: float | None = None
    clock: Callable[[], float] = time.monotonic
    sleep: Callable[[float], None] = time.sleep
    play_started: float | None = None  # clock() value when play began (slot acquired); set by run() if None

    def __post_init__(self) -> None:
        self.history: list[dict[str, Any]] = []
        self.action_count = 0
        self.analysis_step = 0
        self.last_engine_action: GameAction | None = None
        self.token_baseline = 0
        self._transcript_offset = 0

    # --- turn loop ---------------------------------------------------------

    @classmethod
    def from_context(cls, ctx: PlayContextLike, agent: SessionAgent) -> GameSession:
        """Build a session from the runner's PlayContext; the time cap counts from ctx.play_started."""
        return cls(
            game=ctx.game,
            agent=agent,
            state_path=ctx.state_path,
            transcript_path=ctx.transcript_path,
            events_path=ctx.events_path,
            stop_flag=ctx.stop_event,
            time_cap_s=ctx.per_game_cap_s,
            max_actions=ctx.max_actions,
            analyzer_timeout_s=ctx.analyzer_timeout_s,
            deadline_monotonic=ctx.soft_deadline,
            play_started=ctx.play_started,
        )

    def run(self) -> None:
        if self.play_started is None:
            self.play_started = self.clock()
        try:
            self._seed()
            self.analysis_step = 1
            while not self.should_stop():
                if self._observed_state() == GameState.GAME_OVER and self.last_engine_action != GameAction.RESET:
                    self._execute_one(_Request(GameAction.RESET, None, "RESET"), tokens=0, batch_index=1, batch_size=1)
                    continue
                result = self.agent.analyze(
                    state_path=self.state_path,
                    action_count=self.action_count,
                    valid_actions=[a.name for a in self._available() if a != GameAction.RESET],
                    step_env=self.step_env,
                    transcript_path=self.transcript_path,
                    analysis_step=self.analysis_step,
                    request_timeout=self.request_timeout(),
                    should_stop=self.should_stop,
                )
                if result is None:
                    raise SessionCrashed(f"agent returned no result at analysis step {self.analysis_step}")
                self._append_analysis_event()
                if result.get("retryable_failure"):
                    self.sleep(1.0)
                    continue
                if result.get("yielded_control"):
                    continue
                self.analysis_step += 1
        except BaseException as exc:
            if isinstance(exc, Exception):
                self.game.mark_crashed(f"error: {type(exc).__name__}: {exc}")
            raise
        finally:
            if not self.game.solver_note:
                self.game.solver_note = f"tokens={self.agent.generated_tokens}"
            self.game.finish(self._take_tokens())
            self.state_path.unlink(missing_ok=True)

    def should_stop(self) -> bool:
        if self.game.state != "playing" or self.stop_flag.is_set():
            return True
        if self._observed_state() == GameState.WIN:
            return True
        if self.time_cap_s is not None and self.elapsed() >= self.time_cap_s:
            return True
        return self.max_actions is not None and self.action_count >= self.max_actions

    def elapsed(self) -> float:
        return 0.0 if self.play_started is None else self.clock() - self.play_started

    def time_remaining(self) -> float | None:
        return None if self.time_cap_s is None else max(0.0, self.time_cap_s - self.elapsed())

    def request_timeout(self) -> float:
        limits = [self.analyzer_timeout_s]
        remaining = self.time_remaining()
        if remaining is not None:
            limits.append(remaining)
        if self.deadline_monotonic is not None:
            limits.append(self.deadline_monotonic - self.clock())
        return max(0.1, min(limits))

    # --- step_env ----------------------------------------------------------

    def step_env(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        try:
            requests = self._parse(arguments)
        except (TypeError, ValueError) as exc:
            return self._error(str(exc))
        batched = "actions" in arguments

        if self.should_stop() or self._observed_state() == GameState.GAME_OVER:
            return self._terminal()

        payloads: list[dict[str, Any]] = []
        stop_reason: str | None = None
        stop_detail: str | None = None
        for index, request in enumerate(requests, start=1):
            if self.should_stop():
                stop_reason, stop_detail = "stopped", "the run is stopping (stop flag, time cap or action cap)"
                break
            if request.action not in self._available():
                detail = f"{request.display} is not a valid action now; valid: {self._valid_labels()}"
                if not payloads:
                    return self._error(detail)
                stop_reason, stop_detail = "invalid_action", detail
                break
            try:
                payload = self._execute_one(request, self._take_tokens(), index, len(requests))
            except Exception as exc:  # noqa: BLE001 - any engine failure becomes an error payload or action_error (spec 2.3)
                detail = f"{request.display} failed in the engine: {type(exc).__name__}: {exc}"
                if not payloads:
                    return self._error(detail)
                stop_reason, stop_detail = "action_error", detail
                break
            payloads.append(payload)
            for flag in ("run_complete", "game_over", "level_completed"):
                if payload[flag]:
                    stop_reason = flag
                    break
            if stop_reason is not None:
                break

        if not payloads:
            return self._terminal(stop_reason or "stopped", stop_detail)

        result = dict(payloads[-1])
        result.update(
            reward=sum(p["reward"] for p in payloads),
            last_reward=payloads[-1]["reward"],
            batched=batched,
            requested_count=len(requests),
            executed_count=len(payloads),
            requested_actions=[r.display for r in requests],
            executed_actions=[p["action_display"] for p in payloads],
            board_changed=any(p["board_changed"] for p in payloads),
            stopped_early=len(payloads) < len(requests),
            stop_reason=stop_reason,
        )
        if stop_detail is not None:
            result["stop_detail"] = stop_detail
        return result

    def _parse(self, arguments: Mapping[str, Any]) -> list[_Request]:
        has_single = "action" in arguments
        has_batch = "actions" in arguments
        if has_single and has_batch:
            raise ValueError("pass either `action` or `actions`, not both")
        if has_single:
            items: Sequence[Any] = [arguments]
        elif has_batch:
            items = arguments["actions"]
            if not isinstance(items, list) or not items:
                raise TypeError("`actions` must be a non-empty list of {action, row?, col?}")
        else:
            raise ValueError("missing `action` or `actions`")
        return [self._parse_one(index, item) for index, item in enumerate(items)]

    @staticmethod
    def _parse_one(index: int, item: Any) -> _Request:
        if not isinstance(item, Mapping):
            raise TypeError(f"each action must be an object with `action`, got {item!r}")
        name = item.get("action")
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"`action` must be a non-empty string, got {name!r}")
        key = name.strip().upper()
        if key in LABEL_TO_ENGINE:
            action = LABEL_TO_ENGINE[key]
        elif key in ENGINE_NAMES:
            action = ENGINE_NAMES[key]
        else:
            raise ValueError(
                f"action {name!r} (batch position {index}) is not known; use one of {sorted(LABEL_TO_ENGINE)}"
            )
        label = ENGINE_TO_LABEL.get(action, action.name)
        if not action.is_complex():
            return _Request(action, None, label)
        coords = {}
        for axis in ("row", "col"):
            # int() conversion as in the original: floats truncate, numeric strings and bools convert;
            # a missing/None/non-numeric coordinate is an error and nothing runs.
            try:
                value = int(item.get(axis))
            except (TypeError, ValueError):
                raise TypeError(f"{label} needs integer `row` and `col`; got {axis}={item.get(axis)!r}") from None
            coords[axis] = min(GRID_MAX, max(0, value))
        row, col = coords["row"], coords["col"]
        return _Request(action, {"x": col, "y": row}, f"{label}(row={row}, col={col})")

    # --- execution and bookkeeping -----------------------------------------

    def _execute_one(self, request: _Request, tokens: int, batch_index: int, batch_size: int) -> dict[str, Any]:
        before_grid = self._grid()
        before_levels = self._levels_completed()
        self.game.execute(request.action, request.data, generated_tokens=tokens)
        observation = self.game.frame
        self.last_engine_action = request.action
        self.action_count += 1
        state = observation.state
        levels = self._levels_completed()
        run_complete = state == GameState.WIN
        game_over = state == GameState.GAME_OVER
        gained = max(0, levels - before_levels)
        grid = self._grid()
        payload = {
            "executed": True,
            "action_num": self.action_count,
            "level": self._level(),
            "score": levels,
            "reward": gained / self.game.number_of_levels,
            "state": state.name,
            "valid_actions": self._valid_labels(),
            "board_changed": grid != before_grid,
            "done": run_complete,  # WIN only, as in the original; GAME_OVER is recoverable
            "level_completed": gained > 0 and not run_complete,
            "game_over": game_over,
            "run_complete": run_complete,
            "action_name": request.action.name,
            "action_data": request.data,
            "action_display": request.display,
            "batch_index": batch_index,
            "batch_size": batch_size,
            **self._timing(),
        }
        self.history.append({"action": request.display, "frame": self._frame(grid)})
        self._write_state()
        self._append_event(
            {
                "type": "action",
                "title": f"Action {self.action_count}",
                "analysis_step": self.analysis_step or None,
                **{k: payload[k] for k in ("action_num", "level", "score", "state", "reward", "board_changed")},
                **{k: payload[k] for k in ("done", "level_completed", "game_over", "run_complete")},
                **{k: payload[k] for k in ("action_name", "action_display", "batch_index", "batch_size")},
            },
            grid,
        )
        return payload

    def _take_tokens(self) -> int:
        total = self.agent.generated_tokens
        tokens = max(0, total - self.token_baseline)
        self.token_baseline = total
        return tokens

    def _seed(self) -> None:
        grid = self._grid()
        self.history = [{"action": "", "frame": self._frame(grid)}]
        self._write_state()
        self._append_event(
            {
                "type": "initial",
                "title": "Initial State",
                "action_num": 0,
                "analysis_step": None,
                "level": self._level(),
                "score": self._levels_completed(),
                "state": self._observed_state().name,
                "reward": 0.0,
            },
            grid,
        )

    def _write_state(self) -> None:
        document = {"current_frame": self.history[-1]["frame"], "history": self.history}
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.state_path.with_name(self.state_path.name + ".tmp")
        tmp.write_text(json.dumps(document), encoding="utf-8")
        os.replace(tmp, self.state_path)

    def _append_event(self, record: dict[str, Any], grid: list[list[int]]) -> None:
        record = {"board": grid, "board_ascii": _ascii(grid), "run_status": self.game.state, **record}
        self.events_path.parent.mkdir(parents=True, exist_ok=True)
        with self.events_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")

    def _append_analysis_event(self) -> None:
        transcript = ""
        if self.transcript_path.exists():
            with self.transcript_path.open("r", encoding="utf-8") as handle:
                handle.seek(self._transcript_offset)
                transcript = handle.read()
                self._transcript_offset = handle.tell()
        self._append_event(
            {
                "type": "analysis",
                "title": f"Analysis Step {self.analysis_step}",
                "action_num": self.action_count,
                "analysis_step": self.analysis_step,
                "level": self._level(),
                "score": self._levels_completed(),
                "state": self._observed_state().name,
                "transcript": transcript,
            },
            self._grid(),
        )

    # --- payload helpers ---------------------------------------------------

    def _timing(self) -> dict[str, float | None]:
        return {
            "run_elapsed_seconds": round(self.elapsed(), 3),
            "time_remaining_seconds": _round(self.time_remaining()),
        }

    def _error(self, message: str) -> dict[str, Any]:
        return {"executed": False, "error": message, "valid_actions": self._valid_labels(), **self._timing()}

    def _terminal(self, reason: str | None = None, detail: str | None = None) -> dict[str, Any]:
        state = self._observed_state()
        if reason is None:
            if state == GameState.WIN:
                reason, detail = "run_complete", "the game is won; no further actions run"
            elif state == GameState.GAME_OVER:
                reason, detail = "game_over", "GAME_OVER; the harness resets the level before the next turn"
            else:
                reason, detail = "stopped", "the run is stopping (stop flag, time cap or action cap)"
        return {
            "executed": False,
            "action_num": self.action_count,
            "level": self._level(),
            "score": self._levels_completed(),
            "state": state.name,
            "valid_actions": self._valid_labels(),
            "done": state == GameState.WIN,
            "game_over": state == GameState.GAME_OVER,
            "run_complete": state == GameState.WIN,
            "stop_reason": reason,
            "stop_detail": detail,
            **self._timing(),
        }

    # --- views of the game ---------------------------------------------------

    def _observed_state(self) -> GameState:
        return self.game.frame.state

    def _levels_completed(self) -> int:
        return int(self.game.levels_completed)

    def _level(self) -> int:
        return max(1, min(self._levels_completed() + 1, self.game.number_of_levels))

    def _available(self) -> list[GameAction]:
        actions = [GameAction.from_id(int(a)) for a in self.game.frame.available_actions]
        if GameAction.RESET not in actions:
            actions.append(GameAction.RESET)
        return actions

    def _valid_labels(self) -> list[str]:
        return [ENGINE_TO_LABEL.get(a, a.name) for a in self._available() if a != GameAction.RESET]

    def _grid(self) -> list[list[int]]:
        layer = self.game.frame.frame[-1]
        return layer.tolist() if hasattr(layer, "tolist") else [list(row) for row in layer]

    def _frame(self, grid: list[list[int]]) -> dict[str, Any]:
        return {"grid": grid, "step": self.action_count, "level": self._level()}


def _ascii(grid: list[list[int]]) -> str:
    top = len(ASCII_ALPHABET) - 1
    return "\n".join("".join(ASCII_ALPHABET[min(top, max(0, int(v)))] for v in row) for row in grid)


def _round(value: float | None) -> float | None:
    return None if value is None else round(value, 3)


def make_play(
    agent_factory: Callable[[], SessionAgent],
    gate_factory: Callable[[], WorldModelGate | None] | None = None,
) -> Callable[[PlayContextLike], None]:
    """The runner's `play(ctx)` callable: one fresh agent (and optional gate) per game copy.

    The gate lives in the agent only: its `action()` handler does the action accounting
    (observe_level / check_action / record_actions) and it stores and backtests the model
    code (prepend / after_cell). A gate_factory replaces the agent's own gate (ToolAgent
    builds one from PROBE_*).
    """

    def play(ctx: PlayContextLike) -> None:
        agent = agent_factory()
        if gate_factory is not None and hasattr(agent, "gate"):
            agent.gate = gate_factory()
        GameSession.from_context(ctx, agent).run()

    return play
