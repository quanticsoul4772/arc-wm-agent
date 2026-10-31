"""Engine adapter: one place that talks to arc_agi / arcengine (spec 2.1).

`Engine` owns the Arcade (offline or competition) and, in competition mode, a
single scorecard shared by every game. `Game` is one playable copy of a game:
it validates actions, calls the engine, and keeps the bookkeeping the runner
reports (per-level action counts, token and time records, final score).
"""
from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from arcengine import FrameDataRaw, GameAction, GameState

OFFLINE = "offline"
COMPETITION = "competition"

PLAYING = "playing"
WON = "won"
GAVE_UP = "gave_up"
CANCELLED = "cancelled"
CRASHED = "crashed"

GRID_MAX = 63
LEVEL_SCORE_CAP = 115.0
DEFAULT_GATEWAY_URL = "http://gateway:8001/"
# Waits before each retry of a failed scorecard close; arc_agi's close is one 10 s POST with no retry.
CLOSE_RETRY_DELAYS_S = (2.0, 4.0, 8.0, 16.0)


class EngineError(RuntimeError):
    """The engine call failed or returned something unusable."""


class InvalidAction(ValueError):
    """The action or its data is not acceptable in the current state."""


def quiet_logger(name: str = "wmagent.arcade") -> logging.Logger:
    """A private WARNING-level logger for the Arcade, and a quiet scorecard logger."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.WARNING)
    logging.getLogger("arc_agi.scorecard").setLevel(logging.WARNING)
    return logger


def to_game_action(action: GameAction | str | int) -> GameAction:
    if isinstance(action, GameAction):
        return action
    if isinstance(action, bool):
        raise InvalidAction(f"not an action: {action!r}")
    if isinstance(action, int):
        return GameAction.from_id(action)
    if isinstance(action, str):
        return GameAction.from_name(action)
    raise InvalidAction(f"not an action: {action!r}")


def compute_score(baselines: list[int] | None, actions_per_level: list[int], levels_completed: int) -> float:
    """Weighted level score, the same arithmetic as arc_agi's scorecard calculator.

    Level i has weight i+1. A completed level that took at least one action scores
    min(115, (baseline/actions)^2 * 100); every other level scores 0. The weighted
    mean is capped at the share of weight held by levels that scored.
    """
    if not baselines:
        return 0.0
    weighted = 0.0
    weight_sum = 0
    scored_weight = 0
    for index, baseline in enumerate(baselines):
        weight = index + 1
        taken = actions_per_level[index] if index < len(actions_per_level) else 0
        level_score = 0.0
        if index < levels_completed and taken > 0:
            level_score = min(((baseline / taken) ** 2) * 100, LEVEL_SCORE_CAP)
        weighted += level_score * weight
        weight_sum += weight
        if level_score > 0:
            scored_weight += weight
    return min(weighted / weight_sum, scored_weight / weight_sum * 100)


def _task_cancelling() -> bool:
    try:
        task = asyncio.current_task()
    except RuntimeError:
        return False
    return bool(task is not None and task.cancelling())


class SharedScorecard:
    """One scorecard for the whole run, opened on first use and closed once, ref-counted."""

    def __init__(self, arcade: Any) -> None:
        self._arcade = arcade
        self._lock = threading.Lock()
        self._refs = 0
        self.card_id: str | None = None
        self.closed = False

    def acquire(self) -> str:
        with self._lock:
            if self.closed:
                raise EngineError("the shared scorecard is already closed")
            if self.card_id is None:
                self.card_id = str(self._arcade.create_scorecard())
            self._refs += 1
            return self.card_id

    def release(self) -> None:
        with self._lock:
            self._refs -= 1
            if self._refs <= 0:
                self._close_locked()

    def close(self) -> None:
        """Close now regardless of outstanding references (teardown safety net)."""
        with self._lock:
            self._close_locked()

    def _close_locked(self) -> None:
        """Close with retries; `closed` is set only after a call succeeds, so a later close() tries again.
        Raises the last error when every attempt fails."""
        if self.closed or self.card_id is None:
            return
        for attempt, delay in enumerate((0.0, *CLOSE_RETRY_DELAYS_S), 1):
            time.sleep(delay)
            try:
                self._arcade.close_scorecard(self.card_id)
            except Exception as exc:  # noqa: BLE001 - retried, then re-raised below
                print(f"[engine] scorecard close attempt {attempt} failed: {type(exc).__name__}: {exc}", flush=True)
                if getattr(getattr(exc, "response", None), "status_code", None) == 404:
                    if attempt == 1:
                        raise  # the card was gone before this run closed it; a retry cannot help
                    # The server answers 404 "scorecard ... not found" once a card is closed, so an earlier
                    # attempt whose reply was lost (e.g. a timeout) did close it.
                    print("[engine] scorecard already closed by an earlier attempt", flush=True)
                    self.closed = True
                    return
                if attempt == len(CLOSE_RETRY_DELAYS_S) + 1:
                    raise
                continue
            self.closed = True
            return


class Engine:
    """Wraps an arc_agi Arcade in offline or competition mode."""

    def __init__(
        self,
        mode: str = OFFLINE,
        environments_dir: str | None = None,
        base_url: str | None = None,
        api_key: str = "",
        arcade: Any = None,
        stop_event: threading.Event | None = None,
    ) -> None:
        if mode not in (OFFLINE, COMPETITION):
            raise ValueError(f"unknown engine mode {mode!r}")
        self.mode = mode
        self.stop_event = stop_event or threading.Event()
        if arcade is None:
            import arc_agi

            if mode == OFFLINE:
                if not environments_dir:
                    raise ValueError("offline mode needs environments_dir")
                arcade = arc_agi.Arcade(
                    operation_mode=arc_agi.OperationMode.OFFLINE,
                    environments_dir=environments_dir,
                    logger=quiet_logger(),
                )
            else:
                arcade = arc_agi.Arcade(
                    arc_api_key=api_key,
                    operation_mode=arc_agi.OperationMode.COMPETITION,
                    arc_base_url=base_url or os.getenv("ARC_BASE_URL", DEFAULT_GATEWAY_URL),
                    environments_dir="",
                    logger=quiet_logger(),
                )
        self.arcade = arcade
        self.scorecard = SharedScorecard(arcade) if mode == COMPETITION else None
        self._made: set[str] = set()
        self._made_lock = threading.Lock()

    def available_game_ids(self) -> list[str]:
        # the competition gateway can list a game more than once; each id is played once
        return list(dict.fromkeys(env.game_id for env in self.arcade.available_environments))

    def new_game(self, game_id: str) -> Game:
        return Game(self, game_id)

    def _make(self, game_id: str, scorecard_id: str | None) -> Any:
        if self.mode == COMPETITION:
            with self._made_lock:
                if game_id in self._made:
                    raise EngineError(f"{game_id} was already made in this competition run")
                self._made.add(game_id)
            wrapper = self.arcade.make(game_id, scorecard_id=scorecard_id)
        else:
            wrapper = self.arcade.make(game_id)
        # The make-time RESET above must be a full reset; every later RESET restarts the level only.
        os.environ["ONLY_RESET_LEVELS"] = "true"
        if wrapper is None:
            raise EngineError(f"arcade could not make {game_id}")
        return wrapper


@dataclass
class ActionRecord:
    action: GameAction
    data: dict[str, Any]
    generated_tokens: int
    uncached_input_tokens: int
    wallclock_seconds: float

    def to_json(self) -> dict[str, Any]:
        return {
            "action": {"id": self.action.name, "data": dict(self.data)},
            "generated_tokens": self.generated_tokens,
            "uncached_input_tokens": self.uncached_input_tokens,
            "wallclock_seconds": self.wallclock_seconds,
        }


@dataclass
class StepOutcome:
    frame: FrameDataRaw
    level_before: int
    levels_completed: int
    just_won_level: bool
    state: GameState


@dataclass(eq=False)
class Game:
    """One playable copy of a game, from start to finish."""

    engine: Engine
    game_id: str
    label: str = ""
    state: str = PLAYING
    number_of_levels: int = 0
    baseline: list[int] | None = None
    history: list[ActionRecord] = field(default_factory=list)
    actions_per_level: list[int] = field(default_factory=list)
    levels_completed: int = 0
    solver_note: str = ""
    end_reason: str | None = None
    final_generated_tokens: int = 0
    final_uncached_input_tokens: int = 0
    final_wallclock_seconds: float | None = None
    final_score: float | None = None
    started_at: datetime | None = None
    started_mono: float | None = None
    frame: FrameDataRaw | None = None
    last_action: GameAction | None = None

    def __post_init__(self) -> None:
        self._lock = threading.RLock()
        self._wrapper: Any = None
        self._holds_scorecard = False
        self._finished = False

    # ----------------------------------------------------------------- start
    def start(self) -> FrameDataRaw:
        """Make the game in the engine and return the initial observation."""
        with self._lock:
            if self._wrapper is not None:
                raise EngineError(f"{self.game_id} already started")
            card_id = None
            if self.engine.scorecard is not None:
                card_id = self.engine.scorecard.acquire()
                self._holds_scorecard = True
            self.started_at = datetime.now()
            self.started_mono = time.monotonic()
            # On failure the scorecard reference stays held until finish(), so a failed first
            # start cannot close the shared scorecard before the other games are made.
            wrapper = self.engine._make(self.game_id, card_id)
            frame = wrapper.observation_space
            if frame is None or not frame.frame:
                raise EngineError(f"{self.game_id}: no initial frame after make")
            if frame.win_levels <= 0:
                raise EngineError(f"{self.game_id}: win_levels must be > 0, got {frame.win_levels}")
            info = wrapper.environment_info
            self.game_id = info.game_id or self.game_id
            self.number_of_levels = int(frame.win_levels)
            baseline = list(info.baseline_actions or [])
            self.baseline = baseline or None
            self.actions_per_level = [0] * self.number_of_levels
            self.levels_completed = int(frame.levels_completed)
            self.frame = frame
            self._wrapper = wrapper
            return frame

    # ------------------------------------------------------------ properties
    @property
    def engine_state(self) -> GameState | None:
        return None if self.frame is None else self.frame.state

    @property
    def grid(self) -> Any:
        """The visible 64x64 frame: the last layer of the latest observation."""
        return None if self.frame is None else self.frame.frame[-1]

    def valid_actions(self) -> list[str]:
        """Engine names of the currently available actions (RESET is implicitly always valid)."""
        if self.frame is None:
            return []
        return [GameAction.from_id(int(a)).name for a in self.frame.available_actions]

    def is_valid(self, action: GameAction | str | int) -> bool:
        try:
            game_action = to_game_action(action)
        except ValueError:
            return False
        return game_action is GameAction.RESET or game_action.name in self.valid_actions()

    @property
    def generated_tokens(self) -> int:
        return sum(r.generated_tokens for r in self.history) + self.final_generated_tokens

    def elapsed(self) -> float:
        return 0.0 if self.started_mono is None else time.monotonic() - self.started_mono

    # --------------------------------------------------------------- execute
    def _validate(self, action: GameAction, data: dict[str, Any]) -> None:
        if not self.is_valid(action):
            raise InvalidAction(f"{action.name} is not available; valid: {self.valid_actions()}")
        try:
            action.validate_data(data)
        except Exception as exc:
            raise InvalidAction(f"invalid data for {action.name}: {exc}") from exc
        if action.is_simple() and data:
            raise InvalidAction(f"{action.name} takes no data, got {data}")
        if action.is_complex():
            for key in ("x", "y"):
                value = data.get(key)
                if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= GRID_MAX:
                    raise InvalidAction(f"{action.name} needs integer {key} in 0..{GRID_MAX}, got {value!r}")

    def execute(
        self,
        action: GameAction | str | int,
        data: dict[str, Any] | None = None,
        *,
        generated_tokens: int = 0,
        uncached_input_tokens: int = 0,
    ) -> StepOutcome:
        game_action = to_game_action(action)
        payload = dict(data or {})
        with self._lock:
            if self._wrapper is None:
                raise EngineError(f"{self.game_id} has not been started")
            if self.state != PLAYING:
                raise EngineError(f"{self.game_id} is {self.state}; no more actions")
            self._validate(game_action, payload)
            level_before = min(self.levels_completed, self.number_of_levels - 1)
            frame = self._wrapper.step(game_action, data=payload)
            if frame is None:
                raise EngineError(f"{self.game_id}: engine step {game_action.name} failed")
            if not frame.frame:
                raise EngineError(f"{self.game_id}: empty frame list (non-RESET after GAME_OVER?)")
            self.history.append(
                ActionRecord(game_action, payload, int(generated_tokens), int(uncached_input_tokens), self.elapsed())
            )
            self.actions_per_level[level_before] += 1
            previous = self.levels_completed
            self.levels_completed = max(previous, int(frame.levels_completed))
            self.frame = frame
            self.last_action = game_action
            if frame.state == GameState.WIN:
                self.state = WON
            return StepOutcome(frame, level_before, self.levels_completed, self.levels_completed > previous, frame.state)

    # ---------------------------------------------------------------- finish
    def mark_crashed(self, note: str) -> None:
        with self._lock:
            if self.state == PLAYING:
                self.state = CRASHED
                self.solver_note = note

    def finish(self, final_generated_tokens: int = 0, final_uncached_input_tokens: int = 0) -> None:
        """Close the game out once; later calls do nothing."""
        with self._lock:
            if self._finished:
                return
            self._finished = True
            self.final_generated_tokens += int(final_generated_tokens)
            self.final_uncached_input_tokens += int(final_uncached_input_tokens)
            self.final_wallclock_seconds = self.elapsed()
            if self.state == PLAYING:
                stopping = self.engine.stop_event.is_set() or _task_cancelling()
                self.state = CANCELLED if stopping else GAVE_UP
            self.final_score = compute_score(self.baseline, self.actions_per_level, self.levels_completed)
            holds = self._holds_scorecard
            self._holds_scorecard = False
        print(self.finished_line(), flush=True)  # before the release, which can raise if the close fails
        if holds and self.engine.scorecard is not None:
            self.engine.scorecard.release()

    @property
    def finished(self) -> bool:
        return self._finished

    def finished_line(self) -> str:
        bases = self.baseline or []
        per_level = ",".join(
            f"{n}/{bases[i] if i < len(bases) else '-'}" for i, n in enumerate(self.actions_per_level)
        )
        score = self.final_score if self.final_score is not None else 0.0
        return (
            f"[finished] {self.game_id} state={self.state} level={self.levels_completed}/{self.number_of_levels} "
            f"score={score:.2f} actions={len(self.history)} tokens={self.generated_tokens} per-level={per_level}"
            + (f" end={self.end_reason}" if self.end_reason else "")
            + (f" note={self.solver_note!r}" if self.state == CRASHED else "")
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "game_id": self.game_id,
            "number_of_levels": self.number_of_levels,
            "base_actions_per_level": self.baseline,
            "state": self.state,
            "history": [record.to_json() for record in self.history],
            "actions_per_level": list(self.actions_per_level),
            "levels_completed": self.levels_completed,
            "final_score": self.final_score,
            "solver_note": self.solver_note,
            "end_reason": self.end_reason,
            "solver_analysis_html": f"solver_analysis/{self.label}.html" if self.label else None,
            "final_generated_tokens": self.final_generated_tokens,
            "final_uncached_input_tokens": self.final_uncached_input_tokens,
            "final_wallclock_seconds": self.final_wallclock_seconds,
            "started_at": self.started_at.isoformat() if self.started_at else None,
        }
