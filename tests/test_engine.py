"""T1 engine adapter tests (spec 2.1), including the D1 replay of the 30 enf10 runs."""
from __future__ import annotations

import json
import os
import threading
from pathlib import Path

import pytest
from arcengine import GameAction

from tests.t1_fakes import FakeArcade
from wmagent import engine as eng
from wmagent.engine import COMPETITION, OFFLINE, Engine, EngineError, InvalidAction, compute_score

RUNS = Path(os.getenv("WMAGENT_RUNS_DIR", "/runs"))
ENF10 = RUNS / "20261011_165507_probe-enf10" / "kaggle-output" / "benchmark.json"
ENV_FILES = RUNS / "environment_files"


def fake_game(baselines=None, win_levels=3, mode=OFFLINE, stop=None):
    arcade = FakeArcade(["g1-aaaa", "g2-bbbb"], baselines=baselines, win_levels=win_levels)
    engine = Engine(mode, arcade=arcade, stop_event=stop)
    game = engine.new_game("g1-aaaa")
    game.start()
    return engine, game, arcade


# ------------------------------------------------------------------- score
@pytest.mark.parametrize(
    ("baselines", "per_level", "levels", "expected"),
    [
        ([7, 18, 44, 61, 131, 34, 152], [10, 7, 0, 0, 0, 0, 0], 1, 1.7499999999999998),
        ([19, 16, 34, 42, 123, 80, 14, 23, 111], [22, 5, 0, 0, 0, 0, 0, 0, 0], 1, 1.6574839302112028),
        ([17, 38, 31, 16, 41, 60, 26, 159], [8, 8, 0, 0, 0, 0, 0, 0], 1, 2.7777777777777777),
        ([4, 4], [0, 0], 0, 0.0),
        ([4, 4], [1, 2], 2, 100.0),  # level scores 115 and 400->115 capped, total capped at 100
        (None, [3], 1, 0.0),
        ([], [3], 1, 0.0),
    ],
)
def test_compute_score(baselines, per_level, levels, expected):
    assert compute_score(baselines, per_level, levels) == expected


def test_completed_level_with_zero_actions_scores_zero():
    assert compute_score([5, 5], [0, 5], 2) == pytest.approx(200 / 3)
    assert compute_score([5, 5], [0, 0], 1) == 0.0


# --------------------------------------------------------------- execute
def test_start_records_levels_baseline_and_times():
    _, game, _ = fake_game()
    assert game.number_of_levels == 3
    assert game.baseline == [4, 4, 4]
    assert game.started_at is not None and game.started_mono is not None
    assert game.actions_per_level == [0, 0, 0]
    assert os.environ.get("ONLY_RESET_LEVELS") == "true"


def test_empty_baseline_is_none_and_scores_zero():
    _, game, _ = fake_game(baselines=[])
    assert game.baseline is None
    game.execute("ACTION1")
    game.finish()
    assert game.final_score == 0.0


def test_rejects_unavailable_action_and_bad_data_without_bookkeeping():
    _, game, arcade = fake_game()
    with pytest.raises(InvalidAction):
        game.execute(GameAction.ACTION5)  # not in available set
    with pytest.raises(InvalidAction):
        game.execute("ACTION1", {"x": 1})  # simple action with data
    for bad in ({"x": 64, "y": 0}, {"x": -1, "y": 0}, {"x": 3}, {"x": None, "y": 1}, {"x": 1.5, "y": 2}):
        with pytest.raises(InvalidAction):
            game.execute("ACTION6", bad)
    assert game.history == [] and game.actions_per_level == [0, 0, 0]
    assert arcade.wrappers[0].steps == []


def test_reset_is_always_valid_and_counts_as_an_action():
    _, game, _ = fake_game()
    assert "RESET" not in game.valid_actions()
    assert game.is_valid("RESET")
    game.execute(GameAction.RESET)
    assert len(game.history) == 1 and game.actions_per_level == [1, 0, 0]


def test_bookkeeping_level_attribution_and_win():
    _, game, _ = fake_game()
    out = game.execute("ACTION6", {"x": 3, "y": 63}, generated_tokens=50, uncached_input_tokens=7)
    assert not out.just_won_level
    out = game.execute("ACTION1", generated_tokens=5)
    assert out.just_won_level and out.level_before == 0 and game.levels_completed == 1
    game.execute("ACTION1")
    game.execute("ACTION1")
    assert game.state == eng.WON
    assert game.actions_per_level == [2, 1, 1]
    rec = game.history[0].to_json()
    assert rec["action"] == {"id": "ACTION6", "data": {"x": 3, "y": 63}}
    assert rec["generated_tokens"] == 50 and rec["uncached_input_tokens"] == 7
    assert rec["wallclock_seconds"] >= 0
    with pytest.raises(EngineError):
        game.execute("ACTION1")


def test_game_over_is_recoverable_and_empty_frame_is_an_error():
    _, game, _ = fake_game()
    game.execute("ACTION2")
    assert game.engine_state.name == "GAME_OVER" and game.state == eng.PLAYING
    with pytest.raises(EngineError, match="empty frame"):
        game.execute("ACTION3")
    assert len(game.history) == 1
    game.execute("RESET")
    game.execute("ACTION1")
    assert game.levels_completed == 1 and game.last_action is GameAction.ACTION1


def test_finish_is_idempotent_and_prints_line(capsys):
    _, game, _ = fake_game()
    game.execute("ACTION1", generated_tokens=10)
    game.finish(final_generated_tokens=3)
    game.finish(final_generated_tokens=99)
    assert game.state == eng.GAVE_UP
    assert game.final_generated_tokens == 3 and game.generated_tokens == 13
    out = capsys.readouterr().out.strip().splitlines()
    assert out == ["[finished] g1-aaaa state=gave_up level=1/3 score=16.67 actions=1 tokens=13 per-level=1/4,0/4,0/4"]


def test_finish_marks_cancelled_when_stopping_and_keeps_crashed():
    stop = threading.Event()
    _, game, _ = fake_game(stop=stop)
    stop.set()
    game.finish()
    assert game.state == eng.CANCELLED
    _, crashed, _ = fake_game()
    crashed.mark_crashed("error: ValueError: x")
    crashed.finish()
    assert crashed.state == eng.CRASHED and crashed.solver_note == "error: ValueError: x"


def test_competition_scorecard_refcount_and_single_make():
    arcade = FakeArcade(["a-1", "b-2", "c-3"])
    engine = Engine(COMPETITION, arcade=arcade)
    games = [engine.new_game(g) for g in ("a-1", "b-2", "c-3")]
    for g in games:
        g.start()
    assert arcade.created == ["card-0"] and all(card == "card-0" for _, card in arcade.made)
    duplicate = engine.new_game("a-1")
    with pytest.raises(EngineError, match="already made"):
        duplicate.start()
    duplicate.mark_crashed("error: duplicate")
    duplicate.finish()
    games[0].finish()
    games[1].finish()
    games[1].finish()  # idempotent: must not release twice
    assert arcade.closed == []
    games[2].finish()
    assert arcade.closed == ["card-0"]
    engine.scorecard.close()
    assert arcade.closed == ["card-0"]


# ------------------------------------------------------------- D1 replay
@pytest.mark.skipif(not ENF10.exists(), reason="recorded enf10 run data not mounted")
def test_d1_replay_enf10_histories():
    recorded = json.loads(ENF10.read_text(encoding="utf-8"))["game_runs"]
    assert len(recorded) == 30
    engine = Engine(OFFLINE, environments_dir=str(ENV_FILES))
    mismatches = []
    for run in recorded:
        game = engine.new_game(run["game_id"])
        game.start()
        for entry in run["history"]:
            game.execute(entry["action"]["id"], entry["action"]["data"])
        game.finish()
        got = (game.levels_completed, game.actions_per_level, game.final_score)
        want = (run["levels_completed"], run["actions_per_level"], run["final_score"])
        if got != want or game.baseline != run["base_actions_per_level"]:
            mismatches.append((run["solver_analysis_html"], got, want))
    assert mismatches == []
