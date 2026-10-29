"""Tests for scripts/mechanism_baseline.py on small synthetic logs (no arcengine, no network)."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "mechanism_baseline.py"
_spec = importlib.util.spec_from_file_location("mechanism_baseline", SCRIPT)
mb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mb)

CAP = ("World-model gate: at most 2 more action(s) are allowed on this level before an accepted "
       "`predict(ascii, action)` is required. Send a smaller batch.")
BLOCK = "World-model gate: action() is blocked. Define `def predict(ascii, action)` ... Refusal 1 of 8."
PREDICT_CODE = "def predict(ascii, action):\n    return ascii\naction('LEFT')"


def _prompt(level: int, labels: str = "LEFT, RIGHT") -> str:
    return f"\n===== USER PROMPT =====\nStep 1, level {level}. Text.\n\nLabels accepted right now: {labels}. More.\n"


def _call(code: str, result: dict, tokens: int = 100) -> str:
    return (f"\n===== RESPONSE META =====\n{json.dumps({'finish_reason': 'tool_calls', 'generated_tokens': tokens})}\n"
            f"\n===== TOOL CALL =====\npython: {json.dumps({'code': code})}\n"
            f"\n===== TOOL RESULT =====\n{json.dumps(result, indent=2)}\n")


def _error(msg: str) -> dict:
    return {"tool": "python", "error": f"Traceback (most recent call last):\nRuntimeError: {msg}"}


OK = {"tool": "python", "returncode": 0, "stdout": "fine"}


def _game_transcript() -> str:
    return (
        "\n===== SYSTEM PROMPT =====\nsys\n"
        + _prompt(1, "LEFT, MOUSE")
        + _call("action(['LEFT'] * 3)", _error(CAP))                       # cell 0: cap, uncaught
        + _call(PREDICT_CODE, _error(BLOCK))                                # cell 1: block, uncaught, predict
        + _call("print(1)", OK)                                             # cell 2
        + _call("try:\n    action('LEFT')\nexcept RuntimeError as e:\n    print(e)",
                {"tool": "python", "returncode": 0, "stdout": BLOCK})       # cell 3: block, caught
        + _call(PREDICT_CODE.replace("action('LEFT')", "x = 1"), OK)        # cell 4: captured -> accepted
        + _prompt(2, "LEFT, MOUSE")
        + _call("action('LEFT')", OK)                                       # cell 5
    )


GATE = [
    {"event": "session_start", "probe_actions": 8},
    {"event": "batch_capped", "level": 1, "requested": 3, "remaining": 2},
    {"event": "refused", "level": 1, "refusal": 1},
    {"event": "refused", "level": 1, "refusal": 2},
    {"event": "captured", "level": 1, "passed": True},
    {"event": "accepted", "level": 1},
    {"event": "heldout", "level": 1, "confirmed": True},
    {"event": "level_change", "from_level": 1, "to_level": 2},
    {"event": "batch_capped", "level": 2},
    {"event": "refused", "level": 2},
]


def test_parse_transcript_and_m1a_split():
    parsed = mb.parse_transcript(_game_transcript())
    assert len(parsed["calls"]) == 6 and len(parsed["cells"]) == 6
    assert parsed["class"] == "mixed"
    assert [c["level"] for c in parsed["cells"]] == [1, 1, 1, 1, 1, 2]
    out = mb.m1a(parsed["cells"])
    assert out == {"refused_cells": 3, "cap_cells": 1, "block_cells": 2, "define_predict": 1, "caught": 1,
                   "uncaught": 2, "uncaught_define_predict": 1, "cap_uncaught_define_predict": 0,
                   "block_uncaught_define_predict": 1}


def test_game_class():
    assert mb.game_class({"MOUSE"}) == "click"
    assert mb.game_class({"LEFT", "ACTION7"}) == "keys"
    assert mb.game_class({"LEFT", "MOUSE"}) == "mixed"


def test_defines_predict_needs_top_level():
    assert mb.defines_predict(PREDICT_CODE)
    assert not mb.defines_predict("if True:\n    def predict(a, b):\n        return a")
    assert not mb.defines_predict("x = predict('a', 'b')")


def test_alignment_and_blocked_stretch_calls():
    parsed = mb.parse_transcript(_game_transcript())
    mapped, stats = mb.align_gate(parsed["cells"], GATE[:6])
    assert stats["unbound_events"] == 0
    assert mapped[1:6] == [0, 1, 3, 4, 4]
    stretches = mb.m1d(parsed["cells"], parsed["calls"], GATE[:6], mapped)
    assert [(s["outcome"], s["calls"], s["refusals"]) for s in stretches] == [("accepted", 3, 2)]


def test_stretch_open_at_game_end_and_b_replay():
    cells = [{"call": i, "level": 1, "caps": 0, "blocks": int(i == 0), "captures": False, "refused": i == 0,
              "calls_action": True, "kind": "block" if i == 0 else None} for i in range(12)]
    calls = [{"level": 1, "generated_tokens": 10}] * 12
    gate = [{"event": "refused", "level": 1}]
    mapped, _ = mb.align_gate(cells, gate)
    stretches = mb.m1d(cells, calls, gate, mapped)
    assert stretches[0]["outcome"] == "game_end" and stretches[0]["calls"] == 11
    replay = mb.b_replay(stretches, 8)
    assert replay["released_earlier_with_B8"] == 1 and replay["calls_saved_total"] == 3


def test_released_stretch_ends_at_next_acting_cell():
    cells = [{"call": i, "level": 1, "caps": 0, "blocks": int(i in (0, 2)), "captures": False,
              "refused": i in (0, 2), "calls_action": i != 3, "kind": "block" if i in (0, 2) else None}
             for i in range(6)]
    gate = [{"event": "refused", "level": 1}, {"event": "refused", "level": 1},
            {"event": "released_sim_free", "level": 1}]
    mapped, _ = mb.align_gate(cells, gate)
    stretches = mb.m1d(cells, [{}] * 6, gate, mapped)
    assert [(s["outcome"], s["calls"]) for s in stretches] == [("released", 4)]


def test_m1c_counts_refusals_after_confirmed_level():
    out = mb.m1c(GATE)
    assert out["level_changes"] == 1 and out["from_confirmed"] == 1
    assert out["confirmed_next_level_refusals"] == 1 and out["confirmed_next_level_caps"] == 1


def _events(boards_and_displays, level_of=None, batches=None):
    events = [{"type": "initial", "board": [[0]]}]
    for i, (board, display) in enumerate(boards_and_displays):
        index, size = (batches[i] if batches else (1, 1))
        events.append({"type": "action", "board": board, "action_display": display,
                       "level": (level_of[i] if level_of else 1), "level_completed": False,
                       "batch_index": index, "batch_size": size})
    return events


def test_m2a_would_deny_harm_and_reset():
    seq = [([[0]], "LEFT"),   # no change: table {(h0, LEFT)}
           ([[0]], "LEFT"),   # would deny, no change
           ([[1]], "LEFT"),   # board was h0 again: would deny, changed -> harm
           ([[0]], "RESET"),  # clears the table
           ([[0]], "LEFT")]   # not denied after RESET
    out = mb.m2a(mb.action_rows(_events(seq)))
    assert out["actions"] == 5 and out["would_deny"] == 2
    assert out["harm"] == 1 and out["would_deny_no_change"] == 1


def test_m2a_level_change_clears_and_batch_throwaway():
    seq = [([[0]], "UP"), ([[0]], "UP"), ([[0]], "UP"), ([[5]], "DOWN"), ([[0]], "UP")]
    batches = [(1, 1), (1, 3), (2, 3), (3, 3), (1, 1)]
    out = mb.m2a(mb.action_rows(_events(seq, batches=batches)))
    assert out["would_deny"] == 2 and out["checkable_deny"] == 2
    assert out["throwaway_after_any_would_deny"] == 1 and out["throwaway_after_checkable_deny"] == 1
    out = mb.m2a(mb.action_rows(_events([([[0]], "UP"), ([[0]], "UP")], level_of=[1, 2])))
    assert out["would_deny"] == 0


def test_no_change_runs_and_percentile():
    rows = [{"changed": c} for c in (False, False, True, False, True, False, False, False)]
    assert mb.no_change_runs(rows) == ([2, 1], [3])
    assert mb.percentile([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 90) == 9
    assert mb.percentile([], 90) is None


def test_analyze_run_end_to_end(tmp_path):
    run = tmp_path / "run" / "out"
    (run / "transcripts").mkdir(parents=True)
    (run / "artifacts").mkdir()
    (run / "prompts" / "g1_p0").mkdir(parents=True)
    (run / "transcripts" / "g1_p0.txt").write_text(_game_transcript(), encoding="utf-8")
    (run / "artifacts" / "g1_p0_wm_gate.jsonl").write_text("\n".join(json.dumps(e) for e in GATE), encoding="utf-8")
    events = _events([([[1]], "LEFT"), ([[1]], "LEFT")])
    (run / "artifacts" / "g1_p0_events.jsonl").write_text("\n".join(json.dumps(e) for e in events), encoding="utf-8")
    (run / "prompts" / "g1_p0" / "step_0001.json").write_text(
        json.dumps({"message_count": 30, "estimated_tokens": 31000}), encoding="utf-8")
    (run / "benchmark.json").write_text(json.dumps({"game_runs": [{"game_id": "g1", "history": [
        {"wallclock_seconds": 100.0}, {"wallclock_seconds": 160.0}]}]}), encoding="utf-8")
    result = mb.analyze_run(run, 8)
    summary = result["summary"]
    assert summary["m4c"]["calls"] == 6 and summary["m4c"]["actions"] == 2
    assert summary["m4c"]["calls_per_action_pooled"] == 3.0
    assert summary["trims"]["within_5pct_of_limit"] == 1
    assert summary["wallclock"]["s_per_action"]["max"] == 60.0
    assert summary["m2b"]["no_change_runs_final"]["n"] == 1
    assert mb.markdown({"runs": {"r": result}, "pooled": summary}, 8).startswith("# Mechanism baseline")
