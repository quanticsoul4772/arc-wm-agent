"""scripts/read_arm.py stops before computing anything when a run is incomplete, mislabeled or has crashed games."""
import importlib.util
import json
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "read_arm.py"
_spec = importlib.util.spec_from_file_location("read_arm", SCRIPT)
read_arm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(read_arm)

GAMES = ["aa11-1", "bb22-2"]


def _run(tmp_path, name, mechanisms, scores, states=None, levels=None):
    run = tmp_path / name
    (run / "artifacts").mkdir(parents=True)
    game_runs = []
    for i, score in enumerate(scores):
        gid = GAMES[i % 2]
        game_runs.append({"game_id": gid, "final_score": score, "state": (states or {}).get(i, "gave_up"),
                          "levels_completed": (levels or {}).get(i, 1 if score else 0), "solver_note": None})
        start = {"event": "session_start", "mechanisms": mechanisms}
        (run / "artifacts" / f"{gid}_p{i}_wm_gate.jsonl").write_text(json.dumps(start) + "\n")
        rows = [{"step": 1, "turn_request": 1, "completion_tokens": 100},
                {"step": 1, "turn_request": 2, "completion_tokens": 50},
                {"step": 1, "turn_request": 1, "completion_tokens": 300},
                {"step": 2, "turn_request": 1, "completion_tokens": None, "error": "AnalyzerRequestError: timeout"}]
        (run / "artifacts" / f"{gid}_p{i}_requests.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    (run / "benchmark.json").write_text(json.dumps({"game_runs": game_runs}))
    return run


def _main(arm, base, *extra, mech="all-nocont"):
    return read_arm.main(["--arm", str(arm), "--arm-mechanisms", mech, "--base", str(base), "--games", "4",
                          "--draws", "2000", *extra])


def test_a_clear_gain_ships_and_the_economy_is_split_by_turn_type(tmp_path, capsys):
    arm = _run(tmp_path, "arm", "all-nocont", [3.0, 3.0, 3.0, 3.0])
    base = _run(tmp_path, "base", "base", [0.0, 0.0, 0.0, 0.0])
    assert _main(arm, base) == 0
    out = capsys.readouterr().out
    assert "T +3.000" in out and "games with >= 1 level: arm 4  base 0" in out
    assert "after_yield_tokens': 300" in out and "new_step_tokens': 100" in out and "mid_turn_tokens': 50" in out
    assert "failed_calls': 4" in out


def test_a_loss_is_no_ship(tmp_path, capsys):
    arm = _run(tmp_path, "arm", "all-nocont", [0.0, 0.0, 1.0, 0.0])
    base = _run(tmp_path, "base", "base", [1.0, 1.0, 1.0, 1.0])
    assert _main(arm, base) == 0
    assert "=> NO SHIP" in capsys.readouterr().out


def test_missing_benchmark_or_wrong_count_stops(tmp_path, capsys):
    base = _run(tmp_path, "base", "base", [1.0] * 4)
    arm = _run(tmp_path, "arm", "all-nocont", [1.0] * 4)
    (arm / "benchmark.json").unlink()
    assert _main(arm, base) == 1
    assert "no benchmark.json" in capsys.readouterr().err
    short = _run(tmp_path, "short", "all-nocont", [1.0] * 3)
    assert _main(short, base) == 1
    assert "3 games in benchmark.json, expected 4" in capsys.readouterr().err


def test_a_wrong_arm_manifest_stops(tmp_path, capsys):
    arm = _run(tmp_path, "arm", "all", [1.0] * 4)
    base = _run(tmp_path, "base", "base", [1.0] * 4)
    assert _main(arm, base) == 1
    assert "naming 'all-nocont'" in capsys.readouterr().err


def test_a_crash_stops_until_reviewed(tmp_path, capsys):
    arm = _run(tmp_path, "arm", "all-nocont", [1.0] * 4, states={2: "crashed"})
    base = _run(tmp_path, "base", "base", [1.0] * 4)
    assert _main(arm, base) == 1
    assert "crashed or cancelled games" in capsys.readouterr().err
    assert _main(arm, base, "--crashes-reviewed") == 0


def test_a_missing_score_stops(tmp_path, capsys):
    arm = _run(tmp_path, "arm", "all-nocont", [1.0, None, 1.0, 1.0])
    base = _run(tmp_path, "base", "base", [1.0] * 4)
    assert _main(arm, base) == 1
    assert "games without a score" in capsys.readouterr().err
