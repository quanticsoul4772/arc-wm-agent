"""Run recorded model-written python cells through the sandbox and dump the outcomes.

Source: `<run>/**/artifacts/*_events.jsonl` from recorded probe runs. Each
`analysis` event carries the transcript of one agent turn; its python tool calls
are the cells. Each cell runs with the state as of the start of that turn
(history = the initial frame plus every recorded action before the turn). An
`action()` request is answered by replaying the turn's recorded action events in
order; once they run out the request is refused. As in the enforced runs, each
cell runs behind the world-model gate's stored definitions (`gate.prepend`), and
the gate captures and backtests definitions after every cell; the dumped `code`
is the exact sandbox input (`raw_code` is the model's cell). Output: one JSON line per cell
with the cell id, code, stdout, result, error, action requests and the stdout
the original harness recorded (for a self-check).

Usage: python scripts/sandbox_corpus.py RUN_DIR [RUN_DIR ...] --out cells.jsonl [--limit N] [--workers 8]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import tempfile
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wmagent.sandbox import ActionRefused, run_sandboxed_python
from wmagent.world_model_gate import BACKTEST_DRIVER, WorldModelGate

CALL_RE = re.compile(
    r"\[TOOL CALL: python\]\n<tool_call>\n<function=python>\n<parameter=code>\n(.*?)\n</parameter>\n</function>\n</tool_call>\n",
    re.DOTALL,
)
RESULT_HEADER = "[TOOL RESULT: python]\n"
VALID_RE = re.compile(r"Valid actions right now: (.*?)\.\n")
RESULT_KEYS = ("action_num", "level", "score", "reward", "state", "valid_actions", "board_changed", "done",
               "level_completed", "game_over", "run_complete", "action_display")


def _frame(event: dict[str, Any]) -> dict[str, Any]:
    return {"grid": event["board"], "step": event.get("action_num", 0), "level": event.get("level") or 1}


def _compact(event: dict[str, Any], valid: list[str]) -> dict[str, Any]:
    return {"executed": True, **{k: (valid if k == "valid_actions" else event.get(k)) for k in RESULT_KEYS}}


def _recorded_result(transcript: str, end: int) -> str:
    start = transcript.find(RESULT_HEADER, end)
    if start < 0:
        return ""
    start += len(RESULT_HEADER)
    stop = transcript.find("\n[", start)
    stop = len(transcript) if stop < 0 else stop
    return transcript[start:stop].rstrip("\n")


def iter_cells(events_path: Path) -> Iterator[dict[str, Any]]:
    events = [json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    history: list[dict[str, Any]] = []
    last_result: dict[str, Any] = {}
    pending: list[dict[str, Any]] = []
    turn = 0
    for event in events:
        kind = event.get("type")
        if kind == "initial":
            history.append({"action": "", "frame": _frame(event)})
            continue
        if kind == "action":
            pending.append(event)
            continue
        if kind != "analysis":
            continue
        transcript = event.get("transcript") or ""
        found = VALID_RE.search(transcript)
        valid = [a.strip() for a in found.group(1).split(",")] if found else []
        state = {
            "current_frame": history[-1]["frame"] if history else {},
            "history": list(history),
            "valid_actions": valid,
            "last_action_result": dict(last_result),
        }
        for index, match in enumerate(CALL_RE.finditer(transcript)):
            yield {
                "cell_id": f"{events_path.stem.removesuffix('_events')}/t{turn:04d}/c{index:02d}",
                "code": match.group(1),
                "state": state,
                "replay": list(pending),
                "valid": valid,
                "recorded": _recorded_result(transcript, match.end()),
            }
        for action_event in pending:
            history.append({"action": action_event.get("action_display") or "", "frame": _frame(action_event)})
            last_result = _compact(action_event, valid)
        pending = []
        turn += 1


def run_cell(cell: dict[str, Any], timeout: float, code: str) -> tuple[dict[str, Any], dict[str, Any]]:
    queue = list(cell["replay"])
    history = list(cell["state"]["history"])
    requests: list[Any] = []

    def handler(actions: list[dict[str, Any]]) -> dict[str, Any]:
        requests.append(actions)
        if not queue:
            raise ActionRefused("replay: no recorded action left for this turn")
        executed = []
        for _ in actions:
            if not queue:
                break
            event = queue.pop(0)
            executed.append(event)
            history.append({"action": event.get("action_display") or "", "frame": _frame(event)})
            if event.get("done") or event.get("level_completed") or event.get("game_over"):
                break
        result = _compact(executed[-1], cell["valid"])
        result["executed_actions"] = [e.get("action_display") for e in executed]
        state = {"current_frame": history[-1]["frame"], "history": list(history),
                 "valid_actions": cell["valid"], "last_action_result": result}
        return {"action_result": result, "state": state}

    outcome = run_sandboxed_python(code, timeout_seconds=timeout, initial_state=cell["state"], action_handler=handler)
    final_state = {**cell["state"], "history": history, "current_frame": history[-1]["frame"]}
    return final_state, {
        "cell_id": cell["cell_id"],
        "code": code,
        "raw_code": cell["code"],
        "stdout": outcome.get("stdout", ""),
        "result": outcome.get("result"),
        "error": outcome.get("error"),
        "error_type": _error_type(outcome.get("error")),
        "action_requests": requests,
        "recorded": cell["recorded"],
    }


def run_game(events_path: Path, timeout: float, log_dir: Path) -> list[dict[str, Any]]:
    """Run one game's cells in order with the world-model gate's definition store
    prepended, as in the enforced runs; the store evolves through the gate's own
    capture and backtest."""
    gate = WorldModelGate(probe_actions=8, max_refusals=8)
    gate.reset(log_dir / f"{events_path.stem.removesuffix('_events')}_tool_runtime_state.json")
    rows = []
    for cell in iter_cells(events_path):
        gate.observe_level(cell["state"]["current_frame"].get("level"))
        final_state, row = run_cell(cell, timeout, gate.prepend(cell["code"]))
        rows.append(row)

        def backtest(source: str, state: dict[str, Any] = final_state) -> Any:
            def refuse(_: Any) -> Any:
                raise ActionRefused("backtest: no actions")

            out = run_sandboxed_python(source + "\n\n" + BACKTEST_DRIVER, timeout_seconds=timeout,
                                       initial_state=state, action_handler=refuse)
            if out.get("error"):
                gate.note_backtest_error(out["error"])
                return None
            return out["result"]

        gate.after_cell(cell["code"], bool(row["error"]), backtest)
    return rows


def _error_type(error: str | None) -> str | None:
    if not error:
        return None
    last = error.strip().splitlines()[-1]
    return last.split(":", 1)[0] if ":" in last else last


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=30)
    args = parser.parse_args(argv)

    games = [p for run in args.runs for p in sorted(run.rglob("*_events.jsonl"))]
    with tempfile.TemporaryDirectory() as logs, ThreadPoolExecutor(max_workers=args.workers) as pool:
        per_game = list(pool.map(lambda p: run_game(p, args.timeout, Path(logs)), games))
    rows = [row for game in per_game for row in game]
    if args.limit:
        rows = rows[: args.limit]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    harness = [r for r in rows if r["error"] and not r["error"].startswith(("Traceback", "SyntaxError"))]
    print(json.dumps({"cells": len(rows), "user_errors": sum(1 for r in rows if r["error"]) - len(harness),
                      "harness_errors": len(harness),
                      "harness_error_kinds": sorted({r["error"] for r in harness})}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
