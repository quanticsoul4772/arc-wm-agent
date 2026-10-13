import pytest

from wmagent.views import FrameView, HistoryEntryView, TransitionView, install_runtime, normalize_actions


def _frame(fill: int, step: int, level: int = 1) -> dict:
    return {"grid": [[fill, 0], [0, 15]], "step": step, "level": level}


def _state(n_actions: int) -> dict:
    history = [{"action": "", "frame": _frame(0, 0)}]
    for k in range(1, n_actions + 1):
        history.append({"action": f"A{k}", "frame": _frame(k, k)})
    return {
        "current_frame": history[-1]["frame"],
        "history": history,
        "valid_actions": ["UP", "MOUSE"],
        "last_action_result": {"executed": True, "action_num": n_actions},
    }


def test_frame_renders_ascii_from_grid_with_alphabet():
    frame = FrameView({"grid": [[0, 1, 2, 3], [12, 13, 14, 15], [99, -1, 5, 9]], "step": 3, "level": 2})
    assert frame.ascii == "WwgG\nOrNp\npWBb"
    assert frame.shape == (3, 4)
    assert (frame.step, frame.level) == (3, 2)


def test_frame_prefers_given_ascii_and_shape():
    frame = FrameView({"ascii": "ab\ncd", "grid": [], "step": 0, "level": 1, "shape": [2, 2]})
    assert frame.ascii == "ab\ncd"
    assert frame.shape == (2, 2)
    assert [n["color"] for n in frame.segmentation["nodes"]] == ["a", "b", "c", "d"]


def test_segmentation_is_lazy_and_cached():
    frame = FrameView(_frame(5, 0))
    assert frame._segmentation is None
    first = frame.segmentation
    assert frame.segmentation is first


def test_runtime_globals_follow_history():
    ns: dict = {}
    install_runtime(ns, _state(2), lambda actions: ({}, None))
    assert isinstance(ns["history"][0], HistoryEntryView)
    assert [e.action for e in ns["history"]] == ["", "A1", "A2"]
    assert len(ns["transitions"]) == 2
    first, last = ns["transitions"]
    assert isinstance(last, TransitionView)
    assert first.before_frame is ns["history"][0].frame
    assert first.after_frame is ns["history"][1].frame and first.frame is first.after_frame
    assert first.result == {} and last.result == {"executed": True, "action_num": 2}
    assert ns["last_transition"] is last
    assert ns["previous_frame"] is last.before_frame
    assert ns["last_action_frame"] is last.after_frame
    assert ns["last_action"] == "A2"
    assert ns["current_frame"].step == 2 and ns["latest_frame"] is ns["current_frame"]
    assert ns["valid_actions"] == ["UP", "MOUSE"]
    assert ns["result"] is None


def test_seed_only_state_has_no_transitions():
    ns: dict = {}
    install_runtime(ns, _state(0), lambda actions: ({}, None))
    assert ns["transitions"] == [] and ns["last_transition"] is None
    assert ns["previous_frame"] is None and ns["last_action"] is None and ns["last_action_frame"] is None


def test_history_entries_are_not_subscriptable():
    ns: dict = {}
    install_runtime(ns, _state(1), lambda actions: ({}, None))
    with pytest.raises(TypeError):
        ns["history"][0]["action"]


def test_action_refreshes_state_in_place():
    calls = []

    def request(actions):
        calls.append(actions)
        return {"executed": True, "action_num": 3}, _state(3)

    ns: dict = {}
    install_runtime(ns, _state(2), request)
    held_history, held_transitions = ns["history"], ns["transitions"]
    out = ns["action"]("UP")
    assert calls == [[{"action": "UP"}]]
    assert out == {"executed": True, "action_num": 3}
    assert held_history is ns["history"] and len(held_history) == 4
    assert held_transitions is ns["transitions"] and len(held_transitions) == 3
    assert ns["last_action"] == "A3" and ns["current_frame"].step == 3
    # The state carried its own last_action_result, which wins.
    assert ns["last_action_result"] == {"executed": True, "action_num": 3}
    out["executed"] = False
    assert ns["last_action_result"]["executed"] is True


def test_action_result_fills_missing_state_keys():
    state = _state(1)
    state.pop("last_action_result")
    state.pop("valid_actions")
    ns: dict = {}
    install_runtime(ns, _state(0), lambda actions: ({"executed": True, "valid_actions": ["DOWN"]}, state))
    ns["action"](["UP"])
    assert ns["last_action_result"] == {"executed": True, "valid_actions": ["DOWN"]}
    assert ns["valid_actions"] == ["DOWN"]
    # a copy of last_action_result, as in the original
    assert ns["last_transition"].result == ns["last_action_result"]
    assert ns["last_transition"].result is not ns["last_action_result"]


def test_normalize_actions_forms():
    assert normalize_actions("UP") == [{"action": "UP"}]
    assert normalize_actions(["UP", {"action": "MOUSE", "row": 3, "col": 4}]) == [
        {"action": "UP"},
        {"action": "MOUSE", "row": 3, "col": 4},
    ]
    assert normalize_actions({"action": "LEFT", "extra": 1}) == [{"action": "LEFT"}]
    assert normalize_actions(("UP", "DOWN")) == [{"action": "UP"}, {"action": "DOWN"}]


@pytest.mark.parametrize(
    "bad, error",
    [
        ([], ValueError),
        ("", ValueError),
        ({"row": 1}, ValueError),
        ({"action": "MOUSE", "x": 1, "y": 2}, ValueError),
        (5, TypeError),
        ([None], TypeError),
    ],
)
def test_normalize_actions_rejects(bad, error):
    with pytest.raises(error):
        normalize_actions(bad)


def test_xy_rejection_applies_to_the_mouse_label_only():
    """F18: ACTION6 with x/y passes normalisation; step_env then rejects the missing row/col."""
    assert normalize_actions({"action": "ACTION6", "x": 1, "y": 2}) == [{"action": "ACTION6"}]
    with pytest.raises(ValueError):
        normalize_actions({"action": "mouse", "x": 1})


def test_frame_view_repr_and_assignable_ascii():
    """F15: the original's repr format, and `.ascii` can be assigned."""
    frame = FrameView({"grid": [[0] * 4 for _ in range(3)], "step": 1, "level": 2})
    assert repr(frame) == "AsciiFrameView(level=2, step=1, shape=3x4)"
    frame.ascii = "custom"
    assert frame.ascii == "custom"


def test_view_class_names_reprs_and_dir_sets():
    """F15b: class names, str/repr texts and the non-dunder dir() sets of the original's views."""
    ns: dict = {}
    install_runtime(ns, _state(2), lambda actions: ({}, None))
    frame, entry, transition = ns["current_frame"], ns["history"][1], ns["transitions"][0]
    assert [type(x).__name__ for x in (frame, entry, transition)] == ["FrameView", "HistoryEntryView", "TransitionView"]
    frame_text = repr(ns["history"][1].frame)
    assert repr(entry) == str(entry) == f"AsciiHistoryEntryView(action='A1', frame={frame_text})"
    assert repr(transition) == (
        f"ActionTransitionView(action='A1', before_frame={ns['history'][0].frame!r}, after_frame={frame_text})"
    )
    assert str(frame) == repr(frame)

    def public(obj):
        return sorted(name for name in dir(obj) if not name.startswith("__"))

    assert public(frame) == ["_grid", "_segmentation", "ascii", "level", "segmentation", "shape", "step"]
    assert public(entry) == ["action", "frame"]
    assert public(transition) == ["action", "after_frame", "before_frame", "frame", "result"]
    frame.step = 99
    assert frame.step == 99
