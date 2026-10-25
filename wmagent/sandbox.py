"""Run model-written Python in a short-lived, resource-limited child process.

Each call to `run_sandboxed_python` starts a fresh `python -I -S` process (this
file run as a script) in its own session and temp directory. Nothing carries
over between calls. Host and child talk newline-delimited JSON:

    host  -> child  {"code", "timeout_seconds", "sandbox_cwd", "state", "color_chars"}
    child -> host   {"type": "action", "actions": [...]}
    host  -> child  {"type": "action_result", "action_result", "state"} | {"type": "action_error", "error"}
    child -> host   {"type": "final", "stdout", "result", "action_results"}
                  | {"type": "error", "error", "stdout", "action_results"}

The child exposes the state globals from `wmagent.views`, a reduced set of
builtins and an import hook that admits only a fixed list of stdlib modules.
Limits: a wall-clock deadline (the process group is killed on expiry),
RLIMIT_CPU = timeout + 1 s, RLIMIT_FSIZE = 1 MB, RLIMIT_NOFILE = 32 (the child sets these on
itself before any model code runs). POSIX only.
"""
from __future__ import annotations

import json
import os
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

ALLOWED_MODULES = frozenset(
    {
        "bisect",
        "collections",
        "copy",
        "fractions",
        "functools",
        "heapq",
        "itertools",
        "json",
        "math",
        "operator",
        "random",
        "re",
        "statistics",
        "string",
    }
)

# The original's whitelist, exactly 53 names (behavior note F12), plus the guarded __import__. No class
# machinery and only four exception names; NotImplemented and Ellipsis are not names either.
SAFE_BUILTIN_NAMES = (
    "abs", "all", "any", "ascii", "bin", "bool", "bytearray", "bytes", "callable", "chr", "complex", "dict",
    "dir", "divmod", "enumerate", "Exception", "filter", "float", "format", "frozenset", "getattr", "hasattr",
    "hash", "hex", "int", "isinstance", "issubclass", "iter", "len", "list", "map", "max", "min", "next", "oct",
    "ord", "pow", "print", "range", "repr", "reversed", "round", "set", "slice", "sorted", "str", "sum", "tuple",
    "TypeError", "type", "ValueError", "RuntimeError", "zip",
)

CPU_SLACK_SECONDS = 1
FILE_SIZE_LIMIT = 1024 * 1024
OPEN_FILES_LIMIT = 32
CODE_FILENAME = "<python_tool>"


if __name__ != "__main__":
    # The host re-exports the session's refusal type. The child (this file run as a
    # script under -I -S) must not import the session, which needs site-packages.
    # An action handler raises ActionRefused to reject an `action()` request; the
    # model's code sees a RuntimeError carrying the same message.
    from wmagent.session import ActionRefused

__all__ = ["ALLOWED_MODULES", "ActionRefused", "run_sandboxed_python"]


# --------------------------------------------------------------------------
# Host side
# --------------------------------------------------------------------------


def _child_env() -> dict[str, str]:
    return {
        "PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"),
        "HOME": "/tmp",
        "TMPDIR": "/tmp",
        "PYTHONUNBUFFERED": "1",
        "PYTHONIOENCODING": "utf-8",
        "LANG": "C.UTF-8",
    }


def _timeout_label(timeout_seconds: float) -> str:
    return f"{int(timeout_seconds)}s" if float(timeout_seconds).is_integer() else f"{timeout_seconds}s"


class _LineChannel:
    """Reads newline-terminated messages from a pipe fd, honoring a deadline."""

    def __init__(self, fd: int) -> None:
        import selectors

        self.fd = fd
        self.buffer = b""
        self.selector = selectors.DefaultSelector()
        self.selector.register(fd, selectors.EVENT_READ)

    def read_line(self, deadline: float) -> bytes | None:
        """A line (without newline), None at EOF; raises TimeoutError past the deadline."""
        import time

        while b"\n" not in self.buffer:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError
            if not self.selector.select(remaining):
                raise TimeoutError
            chunk = os.read(self.fd, 1 << 16)
            if not chunk:
                return None
            self.buffer += chunk
        line, _, self.buffer = self.buffer.partition(b"\n")
        return line

    def close(self) -> None:
        self.selector.close()


def _kill_group(proc: Any) -> None:
    import signal

    if proc.poll() is None:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    proc.wait()


def run_sandboxed_python(
    code: str,
    timeout_seconds: float = 30,
    initial_state: dict[str, Any] | None = None,
    action_handler: Callable[[list[dict[str, Any]]], dict[str, Any]] | None = None,
    sandbox_cwd: str | os.PathLike | None = None,
    color_chars: str | None = None,
) -> dict[str, Any]:
    """Run `code` in a fresh sandbox process.

    `action_handler(actions)` receives the normalized action list for each
    `action()` call and returns `{"action_result": dict, "state": snapshot}`;
    raising `ActionRefused` (or any exception) rejects the call.

    Returns `{"stdout", "result", "action_results"}` on success, or
    `{"error", "stdout", "action_results"}` on failure.
    """
    import shutil
    import subprocess
    import tempfile
    import time

    from wmagent.segmentation import ASCII_ALPHABET

    action_results: list[Any] = []
    owned_dir = None
    if sandbox_cwd is None:
        owned_dir = tempfile.mkdtemp(prefix="wm_sandbox_")
        sandbox_cwd = owned_dir
    request = {
        "code": code,
        "timeout_seconds": timeout_seconds,
        "sandbox_cwd": str(sandbox_cwd),
        "state": initial_state or {},
        "color_chars": color_chars or ASCII_ALPHABET,
    }
    deadline = time.monotonic() + float(timeout_seconds)
    proc = subprocess.Popen(
        [sys.executable, "-I", "-S", "-u", str(Path(__file__).resolve())],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        cwd=str(sandbox_cwd),
        env=_child_env(),
        start_new_session=True,
    )
    channel = _LineChannel(proc.stdout.fileno())

    def send(message: dict[str, Any]) -> None:
        proc.stdin.write(json.dumps(message).encode("utf-8") + b"\n")
        proc.stdin.flush()

    def failure(error: str) -> dict[str, Any]:
        return {"error": error, "stdout": "", "action_results": action_results}

    try:
        try:
            send(request)
        except (BrokenPipeError, OSError):
            return failure("Sandbox process exited unexpectedly.")
        while True:
            try:
                line = channel.read_line(deadline)
            except TimeoutError:
                return failure(f"Tool timed out after {_timeout_label(timeout_seconds)}")
            if line is None:
                return failure("Sandbox process exited unexpectedly.")
            try:
                message = json.loads(line.decode("utf-8"))
                kind = message["type"]
            except (ValueError, KeyError, TypeError):
                return failure("Sandbox returned an invalid response.")
            if kind == "action":
                reply = _handle_action(action_handler, message.get("actions"), action_results)
                try:
                    send(reply)
                except (BrokenPipeError, OSError):
                    return failure("Sandbox process exited unexpectedly.")
                continue
            if kind == "final":
                return {
                    "stdout": message.get("stdout", ""),
                    "result": message.get("result"),
                    "action_results": message.get("action_results", action_results),
                }
            if kind == "error":
                return {
                    "error": message.get("error", "Unknown error"),
                    "stdout": message.get("stdout", ""),
                    "action_results": message.get("action_results", action_results),
                }
            return failure("Sandbox returned an invalid response.")
    finally:
        _kill_group(proc)
        channel.close()
        for stream in (proc.stdin, proc.stdout):
            try:
                stream.close()
            except OSError:
                pass
        if owned_dir is not None:
            shutil.rmtree(owned_dir, ignore_errors=True)


def _handle_action(
    handler: Callable[[list[dict[str, Any]]], dict[str, Any]] | None,
    actions: Any,
    action_results: list[Any],
) -> dict[str, Any]:
    if handler is None:
        return {"type": "action_error", "error": "action() is not available in this sandbox."}
    try:
        reply = handler(actions)
    except ActionRefused as exc:
        return {"type": "action_error", "error": str(exc)}
    except Exception as exc:  # noqa: BLE001 - the handler's failure is reported to the model's code, not raised here
        return {"type": "action_error", "error": f"{type(exc).__name__}: {exc}"}
    reply = reply if isinstance(reply, dict) else {}
    action_result = reply.get("action_result")
    action_results.append(action_result)
    return {"type": "action_result", "action_result": action_result, "state": reply.get("state")}


# --------------------------------------------------------------------------
# Child side
# --------------------------------------------------------------------------


def _limit_resources(timeout_seconds: float) -> None:
    """Applied by the child to itself before any model code runs."""
    import resource

    cpu = int(timeout_seconds) + CPU_SLACK_SECONDS
    resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu))
    resource.setrlimit(resource.RLIMIT_FSIZE, (FILE_SIZE_LIMIT, FILE_SIZE_LIMIT))
    resource.setrlimit(resource.RLIMIT_NOFILE, (OPEN_FILES_LIMIT, OPEN_FILES_LIMIT))


def _jsonable(value: Any, depth: int = 0) -> Any:
    """JSON-safe copy: sets become lists, inf/nan pass through (JSON Infinity/NaN), other objects use str()."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if depth > 50:
        return str(value)
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_jsonable(v, depth + 1) for v in value]
    if isinstance(value, dict):
        return {k if isinstance(k, str) else str(k): _jsonable(v, depth + 1) for k, v in value.items()}
    return str(value)


def _format_user_error(exc: BaseException) -> str:
    import traceback

    if isinstance(exc, SyntaxError) and exc.filename == CODE_FILENAME:
        return f"SyntaxError: {exc.msg} ({CODE_FILENAME}, line {exc.lineno})"
    frames = [f for f in traceback.extract_tb(exc.__traceback__) if f.filename == CODE_FILENAME]
    lines = ["Traceback (most recent call last):"]
    lines += [f'  File "{CODE_FILENAME}", line {f.lineno}, in {f.name}' for f in frames]
    lines.append("".join(traceback.format_exception_only(type(exc), exc)).rstrip("\n"))
    return "\n".join(lines)


def _safe_builtins() -> dict[str, Any]:
    import builtins

    real_import = builtins.__import__
    table = {name: getattr(builtins, name) for name in SAFE_BUILTIN_NAMES}

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        root = name.split(".")[0]
        if level != 0 or root not in ALLOWED_MODULES:
            # no module list: the original's message is about 130 chars shorter (equivalence report round 8)
            raise ImportError(f"Import of '{name}' is blocked in this sandbox.")
        return real_import(name, globals, locals, fromlist, level)

    table["__import__"] = guarded_import
    return table


def _child_main() -> None:
    import io

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from wmagent.views import install_runtime

    proto_in = sys.stdin.buffer
    proto_out = sys.stdout.buffer

    def send(message: dict[str, Any]) -> None:
        proto_out.write(json.dumps(message).encode("utf-8") + b"\n")
        proto_out.flush()

    def receive() -> dict[str, Any]:
        line = proto_in.readline()
        if not line:
            raise SystemExit(3)
        return json.loads(line.decode("utf-8"))

    request = receive()
    _limit_resources(float(request.get("timeout_seconds") or 30))
    captured = io.StringIO()
    sys.stdout = captured
    sys.stderr = captured
    action_results: list[Any] = []

    def request_actions(actions: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any] | None]:
        send({"type": "action", "actions": actions})
        reply = receive()
        if reply.get("type") == "action_error":
            raise RuntimeError(reply.get("error", "action() failed"))
        action_results.append(reply.get("action_result"))
        return reply.get("action_result"), reply.get("state")

    # No __name__: functions defined by the cell get __module__ None and `__name__` is a NameError,
    # as in the original.
    namespace: dict[str, Any] = {"__builtins__": _safe_builtins()}
    install_runtime(namespace, request.get("state"), request_actions, request.get("color_chars") or "WwgGcBMPRbSYOrNp")

    try:
        compiled = compile(request.get("code", ""), CODE_FILENAME, "exec")
        exec(compiled, namespace)  # noqa: S102 - running the model's code is the point
    except Exception as exc:  # noqa: BLE001 - report every failure of the model's code as text
        # (a BaseException such as SystemExit ends the child; the host reports the generic exit error)
        send({"type": "error", "error": _format_user_error(exc), "stdout": captured.getvalue(),
              "action_results": _jsonable(action_results)})
        return
    send({"type": "final", "stdout": captured.getvalue(), "result": _jsonable(namespace.get("result")),
          "action_results": _jsonable(action_results)})


if __name__ == "__main__":
    _child_main()
