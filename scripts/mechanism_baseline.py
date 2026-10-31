"""Replay recorded run logs and compute the Plan C step-0a mechanism baselines (CPU only, stdlib only).

Source per run directory (the folder holding `transcripts/`, `artifacts/`, `prompts/`, `benchmark.json`):
- `transcripts/<stem>.txt`: the model calls (RESPONSE META), cells (TOOL CALL / TOOL RESULT) and turns
  (USER PROMPT, which names the level);
- `artifacts/<stem>_events.jsonl`: one `action` record per executed action, with the board after it;
- `artifacts/<stem>_wm_gate.jsonl`: the world-model gate's events;
- `prompts/<stem>/step_*.json`: the last request snapshot of each step;
- `benchmark.json`: per-action wallclock.

Metrics (named after the plan's mechanisms):
- M1a: refused cells split by refusal kind, `def predict(` at top level, and caught vs uncaught.
- M1c: level changes out of a level whose held-out check confirmed the model, and the refusals and
  batch caps on the next level before any acceptance.
- M1d: model calls spent in each blocked stretch, and what a release after B calls would change.
- M2a: back-test of the structural no-op deny on the recorded per-action boards.
- M2b: no-change runs, repeated cell code, refusals with no capture following; p90 thresholds.
- M4c: model calls per executed action and generated tokens per call.
- Trims: snapshot message counts and estimated tokens.

Usage: python -I scripts/mechanism_baseline.py RUN_DIR [RUN_DIR ...] --out-dir DIR [--b-calls 8]
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

SECTION_RE = re.compile(r"^===== ([A-Z][A-Z ]*[A-Z]) =====$", re.MULTILINE)
LEVEL_RE = re.compile(r"\blevel (\d+)\b")
LABELS_RE = re.compile(r"Labels accepted right now: ([A-Z0-9_, ]+)\.")
PREDICT_PATTERN = re.compile(r"^def predict\(", re.MULTILINE)
CAP_TEXT = "World-model gate: at most "
BLOCK_TEXT = "World-model gate: action() is blocked"
REFUSAL_TEXTS = (CAP_TEXT, BLOCK_TEXT)
CONTEXT_LIMIT = 32768 - 512 - 512
NEAR_LIMIT_FRACTION = 0.05


# --- small helpers -------------------------------------------------------------


def percentile(values: list[float], q: float) -> float | None:
    """Nearest-rank percentile (q in 0..100); None for an empty list."""
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, -(-len(ordered) * q // 100))
    return ordered[int(rank) - 1]


def distribution(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"n": 0}
    return {
        "n": len(values),
        "min": min(values),
        "p50": percentile(values, 50),
        "p90": percentile(values, 90),
        "max": max(values),
        "mean": round(statistics.fmean(values), 3),
    }


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def board_hash(board: Any) -> str:
    return hashlib.sha1(json.dumps(board, separators=(",", ":")).encode()).hexdigest()


def defines_predict(code: str) -> bool:
    """A top-level `def predict(` the gate would capture (same anchored match plus AST check)."""
    if not PREDICT_PATTERN.search(code):
        return False
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False
    return any(isinstance(node, ast.FunctionDef) and node.name == "predict" for node in tree.body)


def game_class(labels: set[str]) -> str:
    if not labels:
        return "unknown"
    if labels == {"MOUSE"}:
        return "click"
    return "mixed" if "MOUSE" in labels else "keys"


# --- transcript ------------------------------------------------------------------


def parse_sections(text: str) -> list[tuple[str, str]]:
    marks = list(SECTION_RE.finditer(text))
    sections = []
    for i, mark in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(text)
        sections.append((mark.group(1), text[mark.end():end].strip("\n")))
    return sections


def _tool_code(body: str) -> str:
    _, _, args = body.partition(": ")
    try:
        parsed = json.loads(args)
    except json.JSONDecodeError:
        return args
    return str(parsed.get("code", "")) if isinstance(parsed, dict) else args


def _result_error(body: str) -> str | None:
    """The tool result's `error` field, or None when the result has none."""
    try:
        parsed = json.loads(body)
    except json.JSONDecodeError:
        found = re.search(r'"error": "((?:[^"\\]|\\.)*)"', body)
        return found.group(1) if found else None
    if isinstance(parsed, dict) and parsed.get("error"):
        return str(parsed["error"])
    return None


def parse_transcript(text: str) -> dict[str, Any]:
    """Model calls and cells in order. Each cell belongs to the most recent call."""
    calls: list[dict[str, Any]] = []
    cells: list[dict[str, Any]] = []
    labels: set[str] = set()
    level: int | None = None
    pending: dict[str, Any] | None = None
    for name, body in parse_sections(text):
        if name == "USER PROMPT":
            found = LEVEL_RE.search(body)
            if found:
                level = int(found.group(1))
            for match in LABELS_RE.finditer(body):
                labels.update(label.strip() for label in match.group(1).split(",") if label.strip())
        elif name == "RESPONSE META":
            try:
                meta = json.loads(body)
            except json.JSONDecodeError:
                meta = {}
            calls.append({"level": level, "generated_tokens": meta.get("generated_tokens")})
        elif name == "TOOL CALL":
            pending = {"call": len(calls) - 1, "level": level, "code": _tool_code(body).rstrip()}
        elif name == "TOOL RESULT" and pending is not None:
            error = _result_error(body)
            caps = body.count(CAP_TEXT)
            blocks = body.count(BLOCK_TEXT)
            refusal_in_error = error is not None and any(t in error for t in REFUSAL_TEXTS)
            pending.update(
                caps=caps,
                blocks=blocks,
                refused=bool(caps or blocks),
                kind="block" if blocks else ("cap" if caps else None),
                errored=error is not None,
                caught=bool(caps or blocks) and not refusal_in_error,
                defines_predict=defines_predict(pending["code"]),
                calls_action="action(" in pending["code"],
            )
            pending["captures"] = pending["defines_predict"] and not pending["errored"]
            cells.append(pending)
            pending = None
    return {"calls": calls, "cells": cells, "labels": sorted(labels), "class": game_class(labels)}


# --- M1a ---------------------------------------------------------------------------


def m1a(cells: list[dict[str, Any]]) -> dict[str, int]:
    refused = [c for c in cells if c["refused"]]
    out = {
        "refused_cells": len(refused),
        "cap_cells": sum(1 for c in refused if c["kind"] == "cap"),
        "block_cells": sum(1 for c in refused if c["kind"] == "block"),
        "define_predict": sum(1 for c in refused if c["defines_predict"]),
        "caught": sum(1 for c in refused if c["caught"]),
        "uncaught": sum(1 for c in refused if not c["caught"]),
        "uncaught_define_predict": sum(1 for c in refused if not c["caught"] and c["defines_predict"]),
    }
    for kind in ("cap", "block"):
        out[f"{kind}_uncaught_define_predict"] = sum(
            1 for c in refused if c["kind"] == kind and not c["caught"] and c["defines_predict"])
    return out


# --- gate log: M1c and M1d -----------------------------------------------------------


def align_gate(cells: list[dict[str, Any]], gate: list[dict[str, Any]]) -> tuple[list[int | None], dict[str, int]]:
    """Map each gate event to a cell index. Refusal events (refused, batch_capped) bind to cells with
    refusal text, captured events to cells that define predict without an error, in order; every
    other event takes the cell of the latest bound event."""
    want_r = [c["caps"] + c["blocks"] for c in cells]
    want_c = [1 if c["captures"] else 0 for c in cells]
    pos_r = pos_c = 0
    last: int | None = None
    mapped: list[int | None] = []
    unbound = 0
    for event in gate:
        kind = event.get("event")
        if kind in ("refused", "batch_capped"):
            start = pos_r if last is None else max(pos_r, last)
            while start < len(cells) and want_r[start] == 0:
                start += 1
            if start < len(cells):
                want_r[start] -= 1
                pos_r = start
                last = start
            else:
                unbound += 1
        elif kind == "captured":
            start = pos_c if last is None else max(pos_c, last)
            while start < len(cells) and want_c[start] == 0:
                start += 1
            if start < len(cells):
                want_c[start] -= 1
                pos_c = start
                last = start
            else:
                unbound += 1
        mapped.append(last)
    stats = {
        "gate_refusal_events": sum(1 for e in gate if e.get("event") in ("refused", "batch_capped")),
        "transcript_refusal_texts": sum(c["caps"] + c["blocks"] for c in cells),
        "gate_captured_events": sum(1 for e in gate if e.get("event") == "captured"),
        "transcript_capture_cells": sum(1 for c in cells if c["captures"]),
        "unbound_events": unbound,
    }
    return mapped, stats


def m1c(gate: list[dict[str, Any]]) -> dict[str, Any]:
    confirmed_levels: set[Any] = set()
    changes: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for event in gate:
        kind = event.get("event")
        if kind == "heldout" and event.get("confirmed"):
            confirmed_levels.add(event.get("level"))
        if kind == "level_change":
            current = {"from_level": event.get("from_level"), "to_level": event.get("to_level"),
                       "from_confirmed": event.get("from_level") in confirmed_levels,
                       "refusals_before_accept": 0, "caps_before_accept": 0, "accepted": False}
            changes.append(current)
        elif current is not None and not current["accepted"]:
            if kind == "refused":
                current["refusals_before_accept"] += 1
            elif kind == "batch_capped":
                current["caps_before_accept"] += 1
            elif kind == "accepted":
                current["accepted"] = True
    confirmed = [c for c in changes if c["from_confirmed"]]
    return {
        "level_changes": len(changes),
        "from_confirmed": len(confirmed),
        "confirmed_next_level_refusals": sum(c["refusals_before_accept"] for c in confirmed),
        "confirmed_next_level_caps": sum(c["caps_before_accept"] for c in confirmed),
        "confirmed_changes": confirmed,
    }


def m1d(cells: list[dict[str, Any]], calls: list[dict[str, Any]], gate: list[dict[str, Any]],
        mapped: list[int | None]) -> list[dict[str, Any]]:
    """Blocked stretches: from a `refused` event until accepted, released_sim_free, level_change or
    game end. `calls` counts the model calls after the call of the first refused cell, through the
    call whose cell ended the stretch."""
    stretches: list[dict[str, Any]] = []
    open_: dict[str, Any] | None = None
    last_refused_cell: int | None = None

    def close(outcome: str, end_cell: int | None) -> None:
        nonlocal open_
        assert open_ is not None
        start_call = cells[open_["start_cell"]]["call"] if open_["start_cell"] is not None else None
        if end_cell is None or start_call is None:
            open_["calls"] = None
        else:
            end_cell = min(max(end_cell, open_["start_cell"]), len(cells) - 1)
            open_["calls"] = cells[end_cell]["call"] - start_call
        open_["outcome"] = outcome
        stretches.append(open_)
        open_ = None

    for event, cell in zip(gate, mapped):
        kind = event.get("event")
        if kind == "refused":
            last_refused_cell = cell
            if open_ is None:
                open_ = {"level": event.get("level"), "start_cell": cell, "refusals": 0}
            open_["refusals"] += 1
        elif open_ is None:
            continue
        elif kind == "accepted":
            close("accepted", cell)
        elif kind == "released_sim_free":
            start = (last_refused_cell or 0) + 1
            end = next((i for i in range(start, len(cells))
                        if cells[i]["calls_action"] and not cells[i]["refused"]), min(start, len(cells) - 1))
            close("released", end)
        elif kind == "level_change":
            level = open_["level"]
            same = [i for i, c in enumerate(cells) if c["level"] == level and i >= (open_["start_cell"] or 0)]
            close("level_change", same[-1] if same else cell)
    if open_ is not None:
        open_["game_end_calls_total"] = len(calls)
        start_call = cells[open_["start_cell"]]["call"] if open_["start_cell"] is not None else None
        open_["calls"] = None if start_call is None else len(calls) - 1 - start_call
        open_["outcome"] = "game_end"
        stretches.append(open_)
    return stretches


def b_replay(stretches: list[dict[str, Any]], b: int) -> dict[str, Any]:
    known = [s for s in stretches if s["calls"] is not None]
    earlier = [s for s in known if s["calls"] > b]
    by_outcome: dict[str, list[int]] = defaultdict(list)
    for s in known:
        by_outcome[s["outcome"]].append(s["calls"])
    return {
        "stretches": len(stretches),
        "unmapped": len(stretches) - len(known),
        "calls": distribution([s["calls"] for s in known]),
        "calls_by_outcome": {k: distribution(v) for k, v in sorted(by_outcome.items())},
        "outcomes": dict(Counter(s["outcome"] for s in stretches)),
        f"released_earlier_with_B{b}": len(earlier),
        "calls_saved": distribution([s["calls"] - b for s in earlier]),
        "calls_saved_total": sum(s["calls"] - b for s in earlier),
        "earlier_by_outcome": dict(Counter(s["outcome"] for s in earlier)),
    }


# --- events: M2a, M2b ------------------------------------------------------------------


def action_rows(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Executed actions with the board before each one and a per-item change flag."""
    prev_board = None
    rows = []
    for event in events:
        if event.get("type") == "initial":
            prev_board = event.get("board")
            continue
        if event.get("type") != "action":
            continue
        board = event.get("board")
        rows.append({
            "display": event.get("action_display") or event.get("action_name") or "",
            "level": event.get("level"),
            "level_completed": bool(event.get("level_completed")),
            "batch_index": event.get("batch_index") or 1,
            "before": board_hash(prev_board) if prev_board is not None else None,
            "changed": prev_board is None or board != prev_board,
        })
        prev_board = board
    return rows


def m2a(rows: list[dict[str, Any]]) -> dict[str, int]:
    """Table of (board hash before, display) pairs seen with no board change, cleared on RESET and on a
    level change. would_deny: the pair is in the table. checkable: the plan's rule (item 1 of a batch,
    or item k when items 1..k-1 were would-deny no-ops). throwaway: later items of the same batch that
    changed the board, after the first would-deny item (any) or the first checkable deny."""
    table: set[tuple[str | None, str]] = set()
    out = Counter()
    prev_level = None
    prev_completed = False
    batch: list[dict[str, Any]] = []

    def flush() -> None:
        first_any = next((i for i, r in enumerate(batch) if r["would_deny"]), None)
        first_chk = next((i for i, r in enumerate(batch) if r["checkable"]), None)
        if first_any is not None:
            out["throwaway_after_any_would_deny"] += sum(1 for r in batch[first_any + 1:] if r["changed"])
        if first_chk is not None:
            out["throwaway_after_checkable_deny"] += sum(1 for r in batch[first_chk + 1:] if r["changed"])

    for row in rows:
        if row["batch_index"] == 1 and batch:
            flush()
            batch = []
        if row["level"] != prev_level or prev_completed:
            table.clear()
        prev_level, prev_completed = row["level"], row["level_completed"]
        out["actions"] += 1
        is_reset = row["display"].upper().startswith("RESET")
        key = (row["before"], row["display"])
        would = not is_reset and key in table
        checkable = would and all(r["would_deny"] and not r["changed"] for r in batch)
        row = {**row, "would_deny": would, "checkable": checkable}
        if would:
            out["would_deny"] += 1
            out["harm" if row["changed"] else "would_deny_no_change"] += 1
        if checkable:
            out["checkable_deny"] += 1
            out["checkable_harm" if row["changed"] else "checkable_no_change"] += 1
        batch.append(row)
        if is_reset:
            table.clear()
        elif not row["changed"]:
            table.add(key)
    if batch:
        flush()
    for k in ("would_deny", "harm", "would_deny_no_change", "checkable_deny", "checkable_harm",
              "checkable_no_change", "throwaway_after_any_would_deny", "throwaway_after_checkable_deny"):
        out.setdefault(k, 0)
    return dict(out)


def no_change_runs(rows: list[dict[str, Any]]) -> tuple[list[int], list[int]]:
    """Lengths of maximal runs of no-change actions: (runs ended by a change, the final run if any)."""
    closed, final, run = [], [], 0
    for row in rows:
        if row["changed"]:
            if run:
                closed.append(run)
            run = 0
        else:
            run += 1
    if run:
        final.append(run)
    return closed, final


def m2b_cells(cells: list[dict[str, Any]]) -> dict[str, Any]:
    codes = Counter(c["code"].strip() for c in cells if c["code"].strip())
    repeats = {code: n for code, n in codes.items() if n > 1}
    # block refusals with no capture before the next block refusal (or level change / game end)
    blocked = [i for i, c in enumerate(cells) if c["kind"] == "block"]
    no_capture = 0
    streaks: list[int] = []
    streak = 0
    for j, i in enumerate(blocked):
        stop = blocked[j + 1] if j + 1 < len(blocked) else len(cells)
        window = [k for k in range(i, stop) if cells[k]["level"] == cells[i]["level"]]
        next_same = j + 1 < len(blocked) and cells[blocked[j + 1]]["level"] == cells[i]["level"]
        if any(cells[k]["captures"] for k in window):
            if streak:
                streaks.append(streak)
            streak = 0
            continue
        no_capture += 1
        streak += 1
        if not next_same:
            streaks.append(streak)
            streak = 0
    return {
        "repeat_calls": sum(n - 1 for n in repeats.values()),
        "max_repeats": max(codes.values()) if codes else 0,
        "block_refusals": len(blocked),
        "block_refusals_no_capture": no_capture,
        "no_capture_streaks": streaks,
    }


# --- snapshots, benchmark ------------------------------------------------------------------


def snapshots(folder: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(folder.glob("step_*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        rows.append({"message_count": record.get("message_count"), "estimated_tokens": record.get("estimated_tokens")})
    return rows


def wallclock(benchmark: Path) -> list[dict[str, Any]]:
    if not benchmark.exists():
        return []
    data = json.loads(benchmark.read_text(encoding="utf-8"))
    out = []
    for run in data.get("game_runs", []):
        times = [h.get("wallclock_seconds") for h in run.get("history", []) if h.get("wallclock_seconds") is not None]
        if not times:
            out.append({"game_id": run.get("game_id"), "actions": 0})
            continue
        span = times[-1] - times[0]
        out.append({"game_id": run.get("game_id"), "actions": len(times), "span_s": round(span, 1),
                    "s_per_action": round(span / (len(times) - 1), 1) if len(times) > 1 else None})
    return out


# --- per game, per run, pooled ------------------------------------------------------------------


def analyze_game(run_dir: Path, stem: str, b: int) -> dict[str, Any]:
    transcript = parse_transcript((run_dir / "transcripts" / f"{stem}.txt").read_text(encoding="utf-8"))
    cells, calls = transcript["cells"], transcript["calls"]
    gate = read_jsonl(run_dir / "artifacts" / f"{stem}_wm_gate.jsonl")
    events = read_jsonl(run_dir / "artifacts" / f"{stem}_events.jsonl")
    rows = action_rows(events)
    mapped, alignment = align_gate(cells, gate)
    stretches = m1d(cells, calls, gate, mapped)
    closed, final = no_change_runs(rows)
    tokens = [c["generated_tokens"] for c in calls if isinstance(c["generated_tokens"], (int, float))]
    snaps = snapshots(run_dir / "prompts" / stem)
    return {
        "stem": stem,
        "class": transcript["class"],
        "labels": transcript["labels"],
        "calls": len(calls),
        "cells": len(cells),
        "actions": len(rows),
        "generated_tokens": sum(tokens),
        "gen_tokens_per_call": tokens,
        "m1a": m1a(cells),
        "m1c": m1c(gate),
        "m1d_stretches": [{k: s.get(k) for k in ("level", "refusals", "calls", "outcome")} for s in stretches],
        "alignment": alignment,
        "m2a": m2a(rows),
        "no_change_runs_closed": closed,
        "no_change_runs_final": final,
        "m2b": m2b_cells(cells),
        "snapshots": snaps,
    }


def summarize(games: list[dict[str, Any]], b: int, wall: list[dict[str, Any]]) -> dict[str, Any]:
    def sum_keys(key: str) -> dict[str, int]:
        return {k: sum(g[key][k] for g in games) for k in games[0][key]} if games else {}

    m1c_all = [c for g in games for c in g["m1c"]["confirmed_changes"]]
    stretches = [s for g in games for s in g["m1d_stretches"]]
    m2a_by_class: dict[str, Counter] = defaultdict(Counter)
    for g in games:
        m2a_by_class[g["class"]].update(g["m2a"])
    closed = [r for g in games for r in g["no_change_runs_closed"]]
    final = [r for g in games for r in g["no_change_runs_final"]]
    streaks = [s for g in games for s in g["m2b"]["no_capture_streaks"]]
    tokens = [t for g in games for t in g["gen_tokens_per_call"]]
    calls = sum(g["calls"] for g in games)
    actions = sum(g["actions"] for g in games)
    per_game_ratio = [g["calls"] / g["actions"] for g in games if g["actions"]]
    snaps = [s for g in games for s in g["snapshots"]]
    est = [s["estimated_tokens"] for s in snaps if isinstance(s["estimated_tokens"], (int, float))]
    near = sum(1 for t in est if abs(t - CONTEXT_LIMIT) <= NEAR_LIMIT_FRACTION * CONTEXT_LIMIT)
    at_or_above = sum(1 for t in est if t >= (1 - NEAR_LIMIT_FRACTION) * CONTEXT_LIMIT)
    m2b_cells_sum = {k: sum(g["m2b"][k] for g in games) for k in ("repeat_calls", "block_refusals",
                                                                     "block_refusals_no_capture")}
    return {
        "games": len(games),
        "classes": dict(Counter(g["class"] for g in games)),
        "m1a": sum_keys("m1a"),
        "m1c": {
            "level_changes": sum(g["m1c"]["level_changes"] for g in games),
            "from_confirmed": len(m1c_all),
            "next_level_refusals_before_accept": sum(c["refusals_before_accept"] for c in m1c_all),
            "next_level_caps_before_accept": sum(c["caps_before_accept"] for c in m1c_all),
            "next_level_accepted": sum(1 for c in m1c_all if c["accepted"]),
        },
        "m1d": b_replay(stretches, b),
        "alignment": sum_keys("alignment"),
        "m2a_by_class": {k: dict(v) for k, v in sorted(m2a_by_class.items())},
        "m2a_total": dict(sum(m2a_by_class.values(), Counter())),
        "m2b": {
            "no_change_runs_closed": distribution(closed),
            "no_change_runs_final": distribution(final),
            "no_change_run_histogram": dict(sorted(Counter(closed).items())),
            "suggested_no_change_threshold_p90": percentile(closed, 90),
            "repeat_calls": m2b_cells_sum["repeat_calls"],
            "max_repeats_per_game": distribution([g["m2b"]["max_repeats"] for g in games]),
            "suggested_repeat_threshold_p90": percentile([g["m2b"]["max_repeats"] for g in games], 90),
            "block_refusals": m2b_cells_sum["block_refusals"],
            "block_refusals_no_capture": m2b_cells_sum["block_refusals_no_capture"],
            "no_capture_streaks": distribution(streaks),
            "suggested_no_capture_threshold_p90": percentile(streaks, 90),
        },
        "m4c": {
            "calls": calls,
            "actions": actions,
            "calls_per_action_pooled": round(calls / actions, 3) if actions else None,
            "calls_per_action_per_game": distribution([round(r, 3) for r in per_game_ratio]),
            "games_without_actions": sum(1 for g in games if not g["actions"]),
            "gen_tokens_per_call": distribution(tokens),
            "games_le_12_actions": [{"stem": g["stem"], "actions": g["actions"], "calls": g["calls"]}
                                    for g in games if g["actions"] <= 12],
        },
        "trims": {
            "snapshots": len(snaps),
            "message_count": distribution([s["message_count"] for s in snaps if s["message_count"] is not None]),
            "estimated_tokens": distribution(est),
            "limit": CONTEXT_LIMIT,
            "within_5pct_of_limit": near,
            "at_or_above_95pct_of_limit": at_or_above,
        },
        "wallclock": {
            "game_runs": len(wall),
            "span_s": distribution([w["span_s"] for w in wall if w.get("span_s") is not None]),
            "s_per_action": distribution([w["s_per_action"] for w in wall if w.get("s_per_action") is not None]),
        },
    }


def analyze_run(run_dir: Path, b: int) -> dict[str, Any]:
    stems = sorted(p.stem for p in (run_dir / "transcripts").glob("*.txt"))
    games = [analyze_game(run_dir, stem, b) for stem in stems]
    wall = wallclock(run_dir / "benchmark.json")
    gate_start = read_jsonl(run_dir / "artifacts" / f"{stems[0]}_wm_gate.jsonl")[:1] if stems else []
    return {"run_dir": str(run_dir), "gate_config": gate_start[0] if gate_start else None,
            "summary": summarize(games, b, wall), "games": games, "wallclock_runs": wall}


# --- report ----------------------------------------------------------------------------------


def _fmt(d: dict[str, Any] | None) -> str:
    if not d or not d.get("n"):
        return "n=0"
    return f"n={d['n']} p50={d['p50']} p90={d['p90']} max={d['max']} mean={d['mean']}"


def markdown(result: dict[str, Any], b: int) -> str:
    names = list(result["runs"]) + ["pooled"]
    summ = {n: result["runs"][n]["summary"] for n in result["runs"]}
    summ["pooled"] = result["pooled"]
    lines = ["# Mechanism baseline (Plan C step 0a)", "", f"Runs: {', '.join(result['runs'])}. B = {b}.", ""]

    def table(title: str, cols: list[str], getter) -> None:
        lines.extend([f"## {title}", "", "| run | " + " | ".join(cols) + " |",
                      "|---" * (len(cols) + 1) + "|"])
        for n in names:
            lines.append(f"| {n} | " + " | ".join(str(v) for v in getter(summ[n])) + " |")
        lines.append("")

    table("M1a reach (refused cells)", ["refused", "cap", "block", "def predict", "caught", "uncaught",
                                         "uncaught+predict"],
          lambda s: [s["m1a"][k] for k in ("refused_cells", "cap_cells", "block_cells", "define_predict",
                                           "caught", "uncaught", "uncaught_define_predict")])
    table("M1c reach", ["level changes", "from confirmed", "next-level refusals", "next-level caps"],
          lambda s: [s["m1c"][k] for k in ("level_changes", "from_confirmed", "next_level_refusals_before_accept",
                                           "next_level_caps_before_accept")])
    table("M1d blocked stretches (calls after first refusal)", ["stretches", "outcomes", "calls",
                                                                f"earlier with B={b}", "calls saved"],
          lambda s: [s["m1d"]["stretches"], s["m1d"]["outcomes"], _fmt(s["m1d"]["calls"]),
                     s["m1d"][f"released_earlier_with_B{b}"], s["m1d"]["calls_saved_total"]])
    table("M2a no-op deny back-test", ["actions", "would deny", "harm", "checkable", "checkable harm",
                                       "throwaway (any)", "throwaway (checkable)"],
          lambda s: [s["m2a_total"].get(k, 0) for k in ("actions", "would_deny", "harm", "checkable_deny",
                                                        "checkable_harm", "throwaway_after_any_would_deny",
                                                        "throwaway_after_checkable_deny")])
    lines.append("M2a by class (pooled):")
    lines.append("")
    for cls, v in result["pooled"]["m2a_by_class"].items():
        lines.append(f"- {cls}: actions {v.get('actions', 0)}, would deny {v.get('would_deny', 0)}, "
                     f"harm {v.get('harm', 0)}, checkable {v.get('checkable_deny', 0)}, "
                     f"throwaway {v.get('throwaway_after_any_would_deny', 0)}")
    lines.append("")
    table("M2b", ["no-change runs (closed)", "p90 thr", "repeat calls", "max repeats p90",
                  "block refusals no capture", "no-capture streak p90"],
          lambda s: [_fmt(s["m2b"]["no_change_runs_closed"]), s["m2b"]["suggested_no_change_threshold_p90"],
                     s["m2b"]["repeat_calls"], s["m2b"]["suggested_repeat_threshold_p90"],
                     f"{s['m2b']['block_refusals_no_capture']}/{s['m2b']['block_refusals']}",
                     s["m2b"]["suggested_no_capture_threshold_p90"]])
    table("M4c call economy", ["calls", "actions", "calls/action", "per game", "gen tok/call", "games <=12 actions"],
          lambda s: [s["m4c"]["calls"], s["m4c"]["actions"], s["m4c"]["calls_per_action_pooled"],
                     _fmt(s["m4c"]["calls_per_action_per_game"]), _fmt(s["m4c"]["gen_tokens_per_call"]),
                     len(s["m4c"]["games_le_12_actions"])])
    table("Trims (snapshots)", ["snapshots", "message_count", "estimated_tokens", "within 5% of limit"],
          lambda s: [s["trims"]["snapshots"], _fmt(s["trims"]["message_count"]),
                     _fmt(s["trims"]["estimated_tokens"]), s["trims"]["within_5pct_of_limit"]])
    table("Wallclock (benchmark.json, minus first action)", ["game runs", "span s", "s/action"],
          lambda s: [s["wallclock"]["game_runs"], _fmt(s["wallclock"]["span_s"]), _fmt(s["wallclock"]["s_per_action"])])
    table("Gate-log alignment", ["gate refusals", "transcript refusal texts", "gate captured",
                                 "transcript capture cells", "unbound"],
          lambda s: [s["alignment"][k] for k in ("gate_refusal_events", "transcript_refusal_texts",
                                                 "gate_captured_events", "transcript_capture_cells",
                                                 "unbound_events")])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument("--names", nargs="*", help="run names, in the order of RUN_DIR (default: parent folder name)")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--b-calls", type=int, default=8)
    args = parser.parse_args()
    names = args.names or [d.parent.name for d in args.run_dirs]
    runs = {name: analyze_run(d, args.b_calls) for name, d in zip(names, args.run_dirs)}
    all_games = [g for r in runs.values() for g in r["games"]]
    all_wall = [w for r in runs.values() for w in r["wallclock_runs"]]
    result = {"b_calls": args.b_calls, "runs": runs, "pooled": summarize(all_games, args.b_calls, all_wall)}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "baseline.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    (args.out_dir / "summary.md").write_text(markdown(result, args.b_calls), encoding="utf-8")
    print(markdown(result, args.b_calls))


if __name__ == "__main__":
    main()
