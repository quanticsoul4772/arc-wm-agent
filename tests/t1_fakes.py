# ruff: noqa: EXE002
"""Fakes for the T1 engine/runner tests: an in-memory Arcade and environment wrapper."""
from __future__ import annotations

import threading
from dataclasses import dataclass, field

from arcengine import FrameDataRaw, GameAction, GameState


def make_frame(levels: int, win_levels: int, state: GameState = GameState.NOT_FINISHED,
               available: list[int] | None = None, empty: bool = False) -> FrameDataRaw:
    frame = FrameDataRaw()
    frame.state = state
    frame.levels_completed = levels
    frame.win_levels = win_levels
    frame.available_actions = [1, 2, 3, 4, 6] if available is None else available
    frame.frame = [] if empty else [[[levels % 16] * 64 for _ in range(64)]]
    return frame


@dataclass
class FakeInfo:
    game_id: str
    baseline_actions: list[int] = field(default_factory=lambda: [4, 4, 4])


class FakeWrapper:
    """ACTION1 completes the current level; ACTION2 causes GAME_OVER; everything else changes nothing."""

    def __init__(self, info: FakeInfo, win_levels: int = 3) -> None:
        self.environment_info = info
        self.win_levels = win_levels
        self.levels = 0
        self.state = GameState.NOT_FINISHED
        self.steps: list[tuple[GameAction, dict]] = []
        self.observation_space = make_frame(0, win_levels)

    def step(self, action: GameAction, data: dict | None = None) -> FrameDataRaw | None:
        self.steps.append((action, dict(data or {})))
        if action is GameAction.RESET:
            self.state = GameState.NOT_FINISHED
        elif self.state in (GameState.GAME_OVER, GameState.WIN):
            return make_frame(self.levels, self.win_levels, self.state, empty=True)
        elif action is GameAction.ACTION1:
            self.levels += 1
            if self.levels >= self.win_levels:
                self.state = GameState.WIN
        elif action is GameAction.ACTION2:
            self.state = GameState.GAME_OVER
        self.observation_space = make_frame(self.levels, self.win_levels, self.state)
        return self.observation_space


class FakeArcade:
    def __init__(self, game_ids: list[str], baselines: list[int] | None = None, win_levels: int = 3) -> None:
        base = [4, 4, 4] if baselines is None else baselines
        self.available_environments = [FakeInfo(g, list(base)) for g in game_ids]
        self.win_levels = win_levels
        self.lock = threading.Lock()
        self.made: list[tuple[str, str | None]] = []
        self.created: list[str] = []
        self.closed: list[str] = []
        self.wrappers: list[FakeWrapper] = []

    def create_scorecard(self) -> str:
        with self.lock:
            card = f"card-{len(self.created)}"
            self.created.append(card)
            return card

    def close_scorecard(self, card_id: str) -> None:
        with self.lock:
            self.closed.append(card_id)

    def make(self, game_id: str, scorecard_id: str | None = None) -> FakeWrapper | None:
        with self.lock:
            self.made.append((game_id, scorecard_id))
            info = next((e for e in self.available_environments if e.game_id == game_id), None)
            if info is None:
                return None
            wrapper = FakeWrapper(FakeInfo(info.game_id, list(info.baseline_actions)), self.win_levels)
            self.wrappers.append(wrapper)
            return wrapper
