"""Read-only views of the game state handed to sandboxed code.

A state snapshot is a plain dict:

    {"current_frame": frame, "history": [{"action": str, "frame": frame}, ...],
     "valid_actions": [str, ...], "last_action_result": dict}

where each frame is `{"grid": [[int]], "ascii": str?, "step": int, "level": int, "shape": [r, c]?}`.
`ascii` wins when present; otherwise it is rendered from `grid` with the colour
alphabet. `RuntimeState` turns a snapshot into the globals the model's code sees
(`current_frame`, `history`, `transitions`, ...) and refreshes them in place
after each `action()` call. Only the standard library is used.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

from wmagent.segmentation import ASCII_ALPHABET, segment, to_char_rows


class AsciiFrameView:

    def __init__(self, frame: dict[str, Any] | None, color_chars: str = ASCII_ALPHABET) -> None:
        frame = frame or {}
        grid = frame.get("grid") or []
        ascii_board = frame.get("ascii")
        self._color_chars = color_chars
        self._grid = grid
        self._ascii = ascii_board if isinstance(ascii_board, str) else None
        self._segmentation: dict | None = None
        self.step = frame.get("step")
        self.level = frame.get("level")
        shape = frame.get("shape")
        if shape:
            self.shape = tuple(shape)
        elif grid:
            self.shape = (len(grid), len(grid[0]) if grid else 0)
        else:
            rows = self.ascii.split("\n") if self.ascii else []
            self.shape = (len(rows), len(rows[0]) if rows else 0)

    @property
    def ascii(self) -> str:
        if self._ascii is None:
            self._ascii = "\n".join(to_char_rows(self._grid, self._color_chars))
        return self._ascii

    @ascii.setter
    def ascii(self, value: str) -> None:  # assignable, as in the original
        self._ascii = value

    @property
    def segmentation(self) -> dict:
        if self._segmentation is None:
            source = self._grid if self._grid else self.ascii
            self._segmentation = segment(source, self._color_chars)
        return self._segmentation

    def __repr__(self) -> str:
        rows, cols = (tuple(self.shape) + (0, 0))[:2]
        return f"AsciiFrameView(level={self.level}, step={self.step}, shape={rows}x{cols})"


FrameView = AsciiFrameView


class HistoryEntry:

    def __init__(self, action: str, frame: FrameView) -> None:
        self.action = action
        self.frame = frame

    def __repr__(self) -> str:
        return f"HistoryEntry(action={self.action!r}, frame={self.frame!r})"


class TransitionView:

    def __init__(self, action: str, before_frame: FrameView | None, after_frame: FrameView, result: dict) -> None:
        self.action = action
        self.before_frame = before_frame
        self.after_frame = after_frame
        self.result = result

    @property
    def frame(self) -> FrameView:
        return self.after_frame

    def __repr__(self) -> str:
        return f"TransitionView(action={self.action!r}, before={self.before_frame!r}, after={self.after_frame!r})"


STATE_GLOBALS = (
    "current_frame",
    "latest_frame",
    "history",
    "transitions",
    "last_transition",
    "previous_frame",
    "last_action_frame",
    "last_action",
    "valid_actions",
    "last_action_result",
)


def normalize_actions(spec: Any) -> list[dict[str, Any]]:
    """Turn the argument of `action()` into a list of `{action, row?, col?}` dicts."""
    items = list(spec) if isinstance(spec, (list, tuple)) else [spec]
    if not items:
        raise ValueError("action() needs at least one action, e.g. action(['UP']).")
    out: list[dict[str, Any]] = []
    for position, item in enumerate(items):
        if isinstance(item, str):
            label = item.strip()
            if not label:
                raise ValueError(f"action() got an empty action name at index {position}.")
            out.append({"action": label})
        elif isinstance(item, dict):
            label = item.get("action")
            if not isinstance(label, str) or not label.strip():
                raise ValueError(f"action() dict at index {position} needs a non-empty 'action' string.")
            if ("x" in item or "y" in item) and label.strip().upper() == "MOUSE":
                raise ValueError("MOUSE takes 'row' and 'col', not 'x'/'y'.")
            entry: dict[str, Any] = {"action": label.strip()}
            for key in ("row", "col"):
                if key in item:
                    entry[key] = item[key]
            out.append(entry)
        else:
            raise TypeError(f"action() items must be str or dict, got {type(item).__name__} at index {position}.")
    return out


class RuntimeState:
    """Builds the state globals from a snapshot and keeps them current."""

    def __init__(self, namespace: dict[str, Any], color_chars: str = ASCII_ALPHABET) -> None:
        self.namespace = namespace
        self.color_chars = color_chars
        self.history: list[HistoryEntry] = []
        self.transitions: list[TransitionView] = []
        self.valid_actions: list[str] = []
        self.last_action_result: dict[str, Any] = {}

    def load(self, state: dict[str, Any] | None) -> None:
        state = state or {}
        history = [
            HistoryEntry(str(entry.get("action") or ""), FrameView(entry.get("frame"), self.color_chars))
            for entry in state.get("history") or []
        ]
        result = state.get("last_action_result")
        result = dict(result) if isinstance(result, dict) else {}
        transitions: list[TransitionView] = []
        for index, entry in enumerate(history):
            if not entry.action:
                continue
            before = history[index - 1].frame if index > 0 else None
            transitions.append(TransitionView(entry.action, before, entry.frame, {}))
        current = state.get("current_frame")
        current_view = FrameView(current, self.color_chars) if current is not None else (
            history[-1].frame if history else FrameView(None, self.color_chars)
        )
        last = transitions[-1] if transitions else None

        # Refresh in place so references the code already holds stay current.
        self.history[:] = history
        self.transitions[:] = transitions
        self.valid_actions[:] = [str(a) for a in state.get("valid_actions") or []]
        self.last_action_result.clear()
        self.last_action_result.update(result)
        if transitions:
            # a copy, as in the original: not the same object as `last_action_result`
            transitions[-1].result = dict(self.last_action_result)

        ns = self.namespace
        ns["current_frame"] = current_view
        ns["latest_frame"] = current_view
        ns["history"] = self.history
        ns["transitions"] = self.transitions
        ns["last_transition"] = last
        ns["previous_frame"] = last.before_frame if last else None
        ns["last_action_frame"] = last.after_frame if last else None
        ns["last_action"] = last.action if last else None
        ns["valid_actions"] = self.valid_actions
        ns["last_action_result"] = self.last_action_result


def install_runtime(
    namespace: dict[str, Any],
    state: dict[str, Any] | None,
    request_actions: Callable[[list[dict[str, Any]]], tuple[dict[str, Any], dict[str, Any] | None]],
    color_chars: str = ASCII_ALPHABET,
) -> RuntimeState:
    """Populate `namespace` with the state globals and the `action()` function.

    `request_actions(actions)` performs the actions and returns
    `(action_result, new_state_or_None)`; it raises to reject the request.
    """
    runtime = RuntimeState(namespace, color_chars)
    runtime.load(state)

    def action(actions: Any) -> dict[str, Any]:
        normalized = normalize_actions(actions)
        action_result, new_state = request_actions(normalized)
        if isinstance(new_state, dict):
            refreshed = dict(new_state)
            if isinstance(action_result, dict):
                refreshed.setdefault("last_action_result", action_result)
                if "valid_actions" not in refreshed and isinstance(action_result.get("valid_actions"), list):
                    refreshed["valid_actions"] = action_result["valid_actions"]
            runtime.load(refreshed)
        elif isinstance(action_result, dict):
            runtime.last_action_result.clear()
            runtime.last_action_result.update(action_result)
            if runtime.transitions:
                runtime.transitions[-1].result = dict(runtime.last_action_result)
        return dict(action_result) if isinstance(action_result, dict) else action_result

    namespace["action"] = action
    namespace["result"] = None
    return runtime
