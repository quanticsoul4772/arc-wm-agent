import os
import sys
import time
from pathlib import Path

import pytest

from wmagent import sandbox
from wmagent.sandbox import ALLOWED_MODULES, ActionRefused, run_sandboxed_python

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="the sandbox needs POSIX process groups and rlimits")


def _state(n_actions: int = 1) -> dict:
    history = [{"action": "", "frame": {"grid": [[0, 0], [0, 0]], "step": 0, "level": 1}}]
    for k in range(1, n_actions + 1):
        history.append({"action": "UP", "frame": {"grid": [[k % 16, 0], [0, 0]], "step": k, "level": 1}})
    return {"current_frame": history[-1]["frame"], "history": history, "valid_actions": ["UP"], "last_action_result": {}}


def _run(code: str, **kwargs) -> dict:
    kwargs.setdefault("timeout_seconds", 10)
    kwargs.setdefault("initial_state", _state())
    return run_sandboxed_python(code=code, **kwargs)


# --- basics -----------------------------------------------------------------


def test_stdout_and_result():
    out = _run("print('hi')\nresult = {'a': (1, 2), 'b': current_frame.ascii}")
    assert out == {"stdout": "hi\n", "result": {"a": [1, 2], "b": "wW\nWW"}, "action_results": []}


def test_result_defaults_to_none_and_unserialisable_values_become_repr():
    assert _run("x = 1")["result"] is None
    assert _run("result = {1}")["result"] == "{1}"


def test_nothing_persists_between_cells():
    assert not _run("persist_me = 1").get("error")
    out = _run("print(persist_me)")
    assert out["error"].endswith("NameError: name 'persist_me' is not defined")


def test_user_error_traceback_shows_only_tool_frames():
    code = "import json\nprint('before')\ndef f():\n    return json.loads('nope')\nf()\n"
    out = _run(code)
    assert out["stdout"] == "before\n"
    lines = out["error"].splitlines()
    assert lines[0] == "Traceback (most recent call last):"
    assert lines[1] == '  File "<python_tool>", line 5, in <module>'
    assert lines[2] == '  File "<python_tool>", line 4, in f'
    assert lines[3].startswith("json.decoder.JSONDecodeError: Expecting value")
    assert len(lines) == 4


def test_syntax_error_is_reported():
    out = _run("x = (\n")
    assert out["error"].startswith("SyntaxError:") and "<python_tool>" in out["error"]


def test_child_runs_in_given_cwd_with_minimal_env(tmp_path: Path):
    seen = {}

    def handler(actions):
        # The child is blocked on this request, so its /proc entry is readable now.
        pid = _sandbox_child_pid()
        seen["cwd"] = os.readlink(f"/proc/{pid}/cwd")
        seen["env"] = Path(f"/proc/{pid}/environ").read_bytes().split(b"\0")
        return {"action_result": {}, "state": None}

    out = _run("action('UP')", sandbox_cwd=tmp_path, action_handler=handler)
    assert not out.get("error"), out
    assert seen["cwd"] == str(tmp_path)
    env = dict(item.decode().split("=", 1) for item in seen["env"] if item)
    assert env["HOME"] == "/tmp" and env["TMPDIR"] == "/tmp"
    assert not any(key.startswith(("PYTEST", "VIRTUAL_ENV")) for key in env)


# --- limits -----------------------------------------------------------------


def _sandbox_child_pid() -> int:
    script = str(Path(sandbox.__file__).resolve())
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        try:
            argv = Path(f"/proc/{entry}/cmdline").read_bytes().split(b"\0")
        except OSError:
            continue
        if script.encode() in argv and int(entry) != os.getpid():
            return int(entry)
    raise AssertionError("sandbox child not found")


def test_timeout_kills_the_process_group():
    start = time.monotonic()
    out = _run("while True:\n    pass\n", timeout_seconds=2)
    assert time.monotonic() - start < 6
    assert out["error"] == "Tool timed out after 2s"
    with pytest.raises(AssertionError):
        _sandbox_child_pid()


def test_timeout_message_uses_the_configured_seconds():
    out = _run("while True:\n    pass\n", timeout_seconds=1)
    assert out["error"] == "Tool timed out after 1s"


def test_rlimits_are_applied_to_the_child():
    limits = {}

    def handler(actions):
        text = Path(f"/proc/{_sandbox_child_pid()}/limits").read_text()
        for line in text.splitlines():
            for name in ("Max cpu time", "Max file size", "Max open files"):
                if line.startswith(name):
                    limits[name] = line[len(name):].split()[:2]
        return {"action_result": {}, "state": None}

    out = _run("action('UP')", timeout_seconds=7, action_handler=handler)
    assert not out.get("error"), out
    assert limits["Max cpu time"] == ["8", "8"]
    assert limits["Max file size"] == [str(1024 * 1024)] * 2
    assert limits["Max open files"] == ["32", "32"]


def test_unexpected_exit_gives_generic_error(monkeypatch):
    monkeypatch.setattr(sandbox.sys, "executable", "/bin/false")
    out = _run("print(1)")
    assert out["error"] == "Sandbox process exited unexpectedly."


def test_invalid_json_gives_generic_error(monkeypatch):
    monkeypatch.setattr(sandbox.sys, "executable", "/bin/echo")
    out = _run("print(1)")
    assert out["error"] == "Sandbox returned an invalid response."


# --- builtins and imports ---------------------------------------------------


@pytest.mark.parametrize("module", sorted(ALLOWED_MODULES))
def test_whitelisted_modules_import(module):
    out = _run(f"import {module}\nresult = {module}.__name__")
    assert out.get("error") is None and out["result"] == module


def test_from_imports_and_submodules_of_allowed_modules():
    out = _run("from collections import Counter\nimport collections.abc\nresult = Counter('aab')['a']")
    assert out["result"] == 2


@pytest.mark.parametrize("module", ["os", "sys", "subprocess", "inspect", "importlib", "builtins", "io", "wmagent"])
def test_other_imports_are_blocked(module):
    out = _run(f"import {module}")
    assert "ImportError" in out["error"] and module in out["error"]


@pytest.mark.parametrize("name", ["open", "eval", "exec", "compile", "id", "globals", "input", "breakpoint"])
def test_unsafe_builtins_are_missing(name):
    out = _run(f"{name}")
    assert out["error"].endswith(f"NameError: name '{name}' is not defined")


def test_dunder_import_goes_through_the_whitelist():
    assert "ImportError" in _run("__import__('os')")["error"]
    assert _run("result = __import__('math').floor(2.5)")["result"] == 2


def test_common_builtins_and_classes_work():
    code = (
        "class Box:\n    def __init__(self, v):\n        self.v = v\n"
        "try:\n    {}['k']\nexcept KeyError:\n    caught = True\n"
        "result = [sorted(zip('ba', range(2))), isinstance(Box(1), Box), getattr(Box(3), 'v'), caught,"
        " sum(map(abs, [-1, 2])), dir(current_frame)[-1]]"
    )
    out = _run(code)
    assert out.get("error") is None, out
    assert out["result"] == [[["a", 1], ["b", 0]], True, 3, True, 3, "step"]


# --- action protocol --------------------------------------------------------


def _stepper(log: list):
    state = _state(1)

    def handler(actions):
        log.append(actions)
        history = state["history"]
        for a in actions:
            k = len(history)
            history.append({"action": a["action"], "frame": {"grid": [[k, 0], [0, 0]], "step": k, "level": 1}})
        state["current_frame"] = history[-1]["frame"]
        result = {"executed": True, "action_num": len(history) - 1, "executed_count": len(actions)}
        state["last_action_result"] = result
        return {"action_result": result, "state": state}

    return handler


def test_action_forms_are_normalised_and_state_refreshes():
    log: list = []
    code = (
        "r1 = action('UP')\n"
        "r2 = action([{'action': 'MOUSE', 'row': 2, 'col': 3}, 'DOWN'])\n"
        "result = [r1['action_num'], r2['action_num'], len(history), len(transitions), last_action,"
        " current_frame.step, previous_frame.step, last_action_result['executed_count'], transitions[-1].result['action_num']]"
    )
    out = _run(code, action_handler=_stepper(log))
    assert out.get("error") is None, out
    assert log == [[{"action": "UP"}], [{"action": "MOUSE", "row": 2, "col": 3}, {"action": "DOWN"}]]
    assert out["result"] == [2, 4, 5, 4, "DOWN", 4, 3, 2, 4]
    assert [r["action_num"] for r in out["action_results"]] == [2, 4]


def test_refusal_surfaces_as_runtime_error_and_is_catchable():
    def refuse(actions):
        raise ActionRefused("gate says no")

    out = _run("try:\n    action('UP')\nexcept RuntimeError as e:\n    result = str(e)", action_handler=refuse)
    assert out["result"] == "gate says no"
    out = _run("action('UP')", action_handler=refuse)
    assert out["error"].splitlines()[-1] == "RuntimeError: gate says no"
    assert out["action_results"] == []


def test_handler_exception_is_reported_to_the_code():
    def broken(actions):
        raise KeyError("boom")

    out = _run("action('UP')", action_handler=broken)
    assert out["error"].splitlines()[-1].startswith("RuntimeError: KeyError")


def test_bad_action_arguments_never_reach_the_handler():
    log: list = []
    for code in ("action([])", "action({'action': 'MOUSE', 'x': 1, 'y': 2})", "action(3)"):
        out = _run(code, action_handler=_stepper(log))
        assert out["error"].splitlines()[-1].split(":")[0] in ("ValueError", "TypeError")
    assert log == []


def test_no_handler_means_action_fails():
    out = _run("action('UP')")
    assert out["error"].splitlines()[-1].startswith("RuntimeError:")


def test_stdout_and_action_results_survive_a_later_error():
    log: list = []
    out = _run("action('UP')\nprint('moved')\n1 / 0", action_handler=_stepper(log))
    assert out["stdout"] == "moved\n"
    assert len(out["action_results"]) == 1
    assert out["error"].splitlines()[-1] == "ZeroDivisionError: division by zero"


def test_timeout_keeps_host_side_action_results():
    log: list = []
    out = _run("action('UP')\nwhile True:\n    pass", timeout_seconds=2, action_handler=_stepper(log))
    assert out["error"] == "Tool timed out after 2s"
    assert len(out["action_results"]) == 1


# --- recorded corpus ----------------------------------------------------------

CORPUS_RUN = Path(os.environ.get("WM_CORPUS_RUN", "/runs/20261011_165507_probe-enf10/kaggle-output/artifacts"))


@pytest.mark.skipif(not CORPUS_RUN.is_dir(), reason="recorded run data not mounted (set WM_CORPUS_RUN)")
def test_recorded_cells_run_without_harness_errors(tmp_path: Path):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    import json

    import sandbox_corpus

    out = tmp_path / "cells.jsonl"
    assert sandbox_corpus.main([str(CORPUS_RUN), "--out", str(out), "--workers", "16"]) == 0
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert len(rows) >= 500
    harness = [r for r in rows if r["error"] and not r["error"].startswith(("Traceback", "SyntaxError"))]
    assert harness == []
    assert sum(1 for r in rows if not r["error"]) >= 0.8 * len(rows)
    assert all(set(r) >= {"cell_id", "code", "stdout", "result", "error", "action_requests"} for r in rows)
