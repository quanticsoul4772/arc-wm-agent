"""T1 runner tests (spec 2.2): ordering, concurrency, deadlines, isolation, competition, benchmark.json."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from tests.t1_fakes import FakeArcade
from wmagent import engine as eng
from wmagent.engine import COMPETITION, OFFLINE, Engine, quiet_logger
from wmagent.runner import PlayContext, RunConfig, Runner, plan_runs, soft_deadline_epoch

RUNS = Path(os.getenv("WMAGENT_RUNS_DIR", "/runs"))
ENV_FILES = RUNS / "environment_files"
PROBE_GAMES = ["tn36-ef4dde99", "bp35-0a0ad940", "vc33-5430563c", "lp85-305b61c3", "tu93-0768757b", "sb26-7fbdac44"]
ANALYZE_CANDIDATES = [os.getenv("WMAGENT_ANALYZE_PY", ""), "/probe/analyze.py", "D:/Projects/arc-agi3-probe/probe/analyze.py"]
SIX = [f"g{i}-0000000{i}" for i in range(6)]


def config(**kw) -> RunConfig:
    base = {"label": "t1", "game_ids": list(SIX), "n_passes": 5, "concurrency": 10, "max_runtime_s_per_game": None}
    base.update(kw)
    return RunConfig.from_dict(base)


def runner(tmp_path, play, cfg=None, arcade=None, mode=OFFLINE, **kw) -> tuple[Runner, FakeArcade]:
    cfg = cfg or config()
    arcade = arcade or FakeArcade(SIX)
    r = Runner(cfg, play, tmp_path, engine=Engine(mode, arcade=arcade), notebook_start_epoch=time.time(), **kw)
    return r, arcade


# --------------------------------------------------------------- config
def test_plan_runs_is_passes_major_with_counting_stems():
    runs = plan_runs(["a", "b"], 3)
    assert runs == [("a", "a_p0"), ("b", "b_p0"), ("a", "a_p1"), ("b", "b_p1"), ("a", "a_p2"), ("b", "b_p2")]


def test_config_rejects_duplicates_and_unknown_keys():
    with pytest.raises(ValueError, match="duplicate"):
        config(game_ids=["a", "a"])
    with pytest.raises(ValueError, match="unknown"):
        RunConfig.from_dict({"label": "x", "game_ids": ["a"], "bogus": 1})
    assert config(game_ids="competition").competition


def test_soft_deadline_formulas():
    cfg = config(max_runtime_s=32400, soft_deadline_buffer_s=600, submission_margin_s=900)
    assert soft_deadline_epoch(cfg, 1000.0, submission=False) == 1000 + 32400 - 600
    assert soft_deadline_epoch(cfg, 1000.0, submission=True) == 1000 + 32400 - 900
    small = config(max_runtime_s=100, soft_deadline_buffer_s=600)
    assert soft_deadline_epoch(small, 0.0, submission=False) == 50


def test_request_timeout_takes_the_tightest_limit():
    game = Engine(OFFLINE, arcade=FakeArcade(SIX)).new_game(SIX[0])
    now = time.monotonic()
    ctx = PlayContext(game, "s", Path("a"), Path("b"), Path("c"), threading.Event(), per_game_cap_s=50,
                      soft_deadline=now + 20, max_actions=None, analyzer_timeout_s=900, play_started=now)
    assert 19 < ctx.request_timeout() <= 20
    ctx.soft_deadline = now - 5
    assert ctx.request_timeout() == 0.1
    ctx.soft_deadline, ctx.per_game_cap_s = None, None
    assert ctx.request_timeout() == 900


# ---------------------------------------------------------- (b) concurrency
def test_concurrency_cap_and_passes_major_admission(tmp_path):
    admitted: list[str] = []
    releases: dict[str, threading.Event] = {}
    lock = threading.Lock()
    in_flight = [0, 0]  # current, max

    def play(ctx: PlayContext) -> None:
        gate = threading.Event()
        with lock:
            releases[ctx.stem] = gate
            admitted.append(ctx.stem)
            in_flight[0] += 1
            in_flight[1] = max(in_flight[1], in_flight[0])
        ctx.game.execute("ACTION3")
        assert gate.wait(10)
        with lock:
            in_flight[0] -= 1

    r, arcade = runner(tmp_path, play)
    expected = [stem for _, stem in plan_runs(SIX, 5)]
    thread = threading.Thread(target=r.run)
    thread.start()

    def wait_admitted(n):
        deadline = time.monotonic() + 10
        while len(admitted) < n and time.monotonic() < deadline:
            time.sleep(0.005)
        time.sleep(0.02)  # give a wrongly admitted extra game the chance to show up
        return list(admitted)

    assert set(wait_admitted(10)) == set(expected[:10]) and len(admitted) == 10
    assert [g for g, _ in arcade.made] == [stem.rsplit("_p", 1)[0] for stem in expected]  # all made up front
    for n in range(10, 30):
        oldest_running = next(s for s in admitted if not releases[s].is_set())
        releases[oldest_running].set()
        assert wait_admitted(n + 1)[n] == expected[n]
    for gate in releases.values():
        gate.set()
    thread.join(10)
    assert in_flight[1] == 10
    assert admitted[10:] == expected[10:]
    assert {g.state for g in r.games} == {eng.GAVE_UP}


# ------------------------------------------------------------- (c) deadline
def test_soft_deadline_cancels_unfinished_games_within_drain(tmp_path):
    def play(ctx: PlayContext) -> None:
        if ctx.stem == "g0-00000000_p0":
            time.sleep(4)  # ignores the stop flag; must be force-finished after the drain
            return
        while not ctx.should_stop():
            time.sleep(0.01)

    cfg = config(max_runtime_s=2.0, cancel_drain_timeout_s=0.5)  # deadline = start + 1 s
    r, _ = runner(tmp_path, play, cfg=cfg)
    t0 = time.monotonic()
    result = r.run()
    elapsed = time.monotonic() - t0
    assert elapsed < 1.0 + 0.5 + 1.0
    assert [g.state for g in result.games] == [eng.CANCELLED] * 30
    data = json.loads((tmp_path / "benchmark.json").read_text())
    assert [g["state"] for g in data["game_runs"]] == ["cancelled"] * 30


def test_per_game_cap_finishes_as_gave_up_and_counts_from_play_start(tmp_path):
    seen = []

    def play(ctx: PlayContext) -> None:
        seen.append(ctx.per_game_cap_s)
        while not ctx.should_stop():
            time.sleep(0.01)
        assert ctx.time_cap_reached()

    cfg = config(game_ids=SIX[:2], n_passes=2, concurrency=2, max_runtime_s_per_game=0.2)
    r, _ = runner(tmp_path, play, cfg=cfg)
    t0 = time.monotonic()
    result = r.run()
    assert time.monotonic() - t0 >= 0.4  # two waves of 0.2 s each: cap measured from slot acquisition
    assert [g.state for g in result.games] == [eng.GAVE_UP] * 4 and seen == [0.2] * 4


def test_action_cap_stops_play(tmp_path):
    def play(ctx: PlayContext) -> None:
        while not ctx.should_stop():
            ctx.game.execute("ACTION3")

    r, _ = runner(tmp_path, play, cfg=config(game_ids=SIX[:1], n_passes=1, max_actions_per_game=7))
    assert len(r.run().games[0].history) == 7


# ------------------------------------------------------------ isolation
def test_error_in_one_game_crashes_only_that_game(tmp_path, capsys):
    def play(ctx: PlayContext) -> None:
        ctx.game.execute("ACTION1", generated_tokens=11)
        if ctx.stem == "g2-00000002_p1":
            raise ValueError("boom")
        ctx.game.solver_note = "tokens=11"

    r, _ = runner(tmp_path, play)
    games = r.run().games
    crashed = [g for g in games if g.state == eng.CRASHED]
    assert [g.label for g in crashed] == ["g2-00000002_p1"]
    assert crashed[0].solver_note == "error: ValueError: boom"
    assert sum(g.state == eng.GAVE_UP for g in games) == 29
    out = capsys.readouterr().out
    assert out.count("[finished] ") == 30 and "runs: 30" in out and "total tokens: 330" in out


def test_start_failure_is_isolated(tmp_path):
    arcade = FakeArcade(SIX[:2])
    r, _ = runner(tmp_path, lambda ctx: None, cfg=config(game_ids=[SIX[0], "zz99-missing"], n_passes=1), arcade=arcade)
    games = r.run().games
    assert games[0].state == eng.GAVE_UP
    assert games[1].state == eng.CRASHED and games[1].solver_note.startswith("error: EngineError")


# ------------------------------------------------- competition (fake gateway)
def test_competition_one_scorecard_closed_after_last_finish(tmp_path):
    arcade = FakeArcade(SIX, baselines=[])
    closed_before_finish = []

    def play(ctx: PlayContext) -> None:
        closed_before_finish.append(bool(arcade.closed))
        ctx.game.execute("ACTION1")

    r, _ = runner(tmp_path, play, cfg=config(game_ids="competition"), arcade=arcade, mode=COMPETITION)
    result = r.run()
    assert arcade.created == ["card-0"] and arcade.closed == ["card-0"]
    assert sorted(g for g, _ in arcade.made) == sorted(SIX) and len(arcade.made) == 6
    assert not any(closed_before_finish) and len(result.games) == 6
    assert all(g.baseline is None and g.final_score == 0.0 for g in result.games)


def test_competition_close_failing_at_last_finish_is_retried_at_teardown(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(eng, "CLOSE_RETRY_DELAYS_S", (0.0, 0.0, 0.0, 0.0))
    calls = []

    class FailingFirstClose(FakeArcade):
        def close_scorecard(self, card_id):
            calls.append(card_id)
            if len(calls) <= 5:  # every attempt made from the last game's release fails
                raise ConnectionError("gateway timeout")
            super().close_scorecard(card_id)

    arcade = FailingFirstClose(SIX, baselines=[])
    r, _ = runner(tmp_path, lambda ctx: ctx.game.execute("ACTION1"), cfg=config(game_ids="competition"),
                  arcade=arcade, mode=COMPETITION)
    result = r.run()
    assert len(calls) == 6 and arcade.closed == ["card-0"] and r.engine.scorecard.closed
    assert len(result.games) == 6
    assert "ConnectionError: gateway timeout" in capsys.readouterr().out  # the worker's error is reported


def test_submission_mode_derives_cap_skips_benchmark_and_closes_before_deadline(tmp_path):
    arcade = FakeArcade(SIX)

    def play(ctx: PlayContext) -> None:
        while not ctx.should_stop():
            time.sleep(0.01)

    # Notebook started 32400 - 900 - 1.2 s ago: 1.2 s left before the submission soft deadline.
    cfg = config(game_ids="competition", max_runtime_s_per_game=2700, concurrency=3, cancel_drain_timeout_s=0.5)
    start = time.time() - (32400 - 900 - 1.2)
    r = Runner(cfg, play, tmp_path, engine=Engine(COMPETITION, arcade=arcade), submission=True,
               notebook_start_epoch=start)
    t0 = time.monotonic()
    result = r.run()
    assert time.monotonic() - t0 < 2.5
    assert result.benchmark_path is None and not (tmp_path / "benchmark.json").exists()
    assert arcade.closed == ["card-0"]
    caps = {round(ctx.per_game_cap_s, 1) for ctx in r.contexts.values()}
    assert caps and max(caps) <= 0.6  # (1.2 s remaining) / ceil(6/3) waves
    assert {g.state for g in result.games} <= {eng.GAVE_UP, eng.CANCELLED}


# ------------------------------------ (d) competition against arc_agi's server
@pytest.fixture
def competition_gateway(tmp_path):
    flask = pytest.importorskip("flask")  # noqa: F841
    from arc_agi import Arcade, OperationMode
    from arc_agi.server import create_app
    from werkzeug.serving import make_server

    if not ENV_FILES.exists():
        pytest.skip("environment files not mounted")
    offline = Arcade(operation_mode=OperationMode.OFFLINE, environments_dir=str(ENV_FILES),
                     recordings_dir=str(tmp_path / "server_rec"), logger=quiet_logger("t1.gateway"))
    app, api = create_app(offline, competition_mode=True)
    counts = {"open": 0, "close": 0}
    new_scorecard, close_scorecard = api.new_scorecard, api.close_scorecard

    def counted_open():
        counts["open"] += 1
        return new_scorecard()

    def counted_close():
        counts["close"] += 1
        return close_scorecard()

    api.new_scorecard, api.close_scorecard = counted_open, counted_close
    server = make_server("127.0.0.1", 0, app, threaded=True)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}", counts
    server.shutdown()


def test_d_competition_mode_against_arc_agi_server(tmp_path, competition_gateway, monkeypatch):
    url, counts = competition_gateway
    monkeypatch.setenv("RECORDINGS_DIR", str(tmp_path / "client_rec"))
    engine = Engine(COMPETITION, base_url=url, api_key="test-key-123")
    made: list[str] = []
    real_make = engine.arcade.make

    def spy_make(game_id, **kw):
        made.append(game_id)
        return real_make(game_id, **kw)

    engine.arcade.make = spy_make

    def play(ctx: PlayContext) -> None:
        names = [a for a in ctx.game.valid_actions() if a != "RESET"]
        for i in range(3):
            name = names[i % len(names)]
            data = {"x": 10 + i, "y": 20} if name == "ACTION6" else None
            ctx.game.execute(name, data, generated_tokens=1)

    cfg = config(game_ids="competition", concurrency=3)
    r = Runner(cfg, play, tmp_path / "out", engine=engine, submission=True, notebook_start_epoch=time.time())
    result = r.run()
    assert counts == {"open": 1, "close": 1}
    assert sorted(made) == sorted(PROBE_GAMES)
    assert all(g.state == eng.GAVE_UP and g.baseline is None and len(g.history) == 3 for g in result.games)


# ---------------------------------------------- (e) benchmark.json in analyze.py
def test_e_benchmark_json_loads_in_analyze(tmp_path):
    analyze = next((Path(p) for p in ANALYZE_CANDIDATES if p and Path(p).exists()), None)
    if analyze is None:
        pytest.skip("probe/analyze.py not available")
    if not ENV_FILES.exists():
        pytest.skip("environment files not mounted")

    def play(ctx: PlayContext) -> None:
        name = ctx.game.valid_actions()[0]
        ctx.game.execute(name, {"x": 5, "y": 5} if name == "ACTION6" else None, generated_tokens=42)
        ctx.game.solver_note = "tokens=42"

    cfg = config(game_ids=PROBE_GAMES[:2], n_passes=2, concurrency=2, environments_dir=str(ENV_FILES))
    result = Runner(cfg, play, tmp_path, notebook_start_epoch=time.time()).run()
    data = json.loads(result.benchmark_path.read_text())
    assert set(data) >= {"label", "n_passes", "solver_label", "start_time", "end_time", "game_runs"}
    run_keys = {"game_id", "number_of_levels", "base_actions_per_level", "state", "history", "actions_per_level",
                "levels_completed", "final_score", "solver_note", "final_generated_tokens",
                "final_wallclock_seconds", "started_at", "solver_analysis_html"}
    assert all(set(g) >= run_keys for g in data["game_runs"])
    proc = subprocess.run([sys.executable, "-I", str(analyze), str(tmp_path)], capture_output=True, text=True, check=False)
    assert proc.returncode == 0, proc.stderr
    rows = [json.loads(line) for line in proc.stdout.splitlines() if line.startswith('{"run"')]
    assert [row["stem"] for row in rows] == [f"{PROBE_GAMES[0]}_p0", f"{PROBE_GAMES[1]}_p0",
                                             f"{PROBE_GAMES[0]}_p1", f"{PROBE_GAMES[1]}_p1"]
    assert all(row["tokens"] == 42 and row["actions"] == 1 and not row["crashed"] for row in rows)


# ------------------------------------------------- idle guard: make first, keep alive, then play
def test_ready_runs_after_makes_and_keepalive_resets_until_it_returns(tmp_path, monkeypatch, capsys):
    import wmagent.runner as rmod
    monkeypatch.setattr(rmod, "KEEPALIVE_S", 0.05)
    arcade = FakeArcade(SIX, baselines=[])
    order = []

    def play(ctx: PlayContext) -> None:
        order.append(("play", len(arcade.made)))
        ctx.game.execute("ACTION1")

    def ready() -> None:
        order.append(("ready", len(arcade.made)))
        time.sleep(0.4)

    r, _ = runner(tmp_path, play, cfg=config(game_ids="competition"), arcade=arcade, mode=COMPETITION)
    result = r.run(ready)
    assert order[0] == ("ready", 6) and all(o[0] == "play" for o in order[1:]) and len(order) == 7
    kept = [g for g in result.games if any(s.action.name == "RESET" for s in g.history)]
    assert len(kept) == 1 and sum(1 for s in kept[0].history if s.action.name == "RESET") >= 3
    assert arcade.created == ["card-0"] and arcade.closed == ["card-0"] and len(arcade.made) == 6
    assert "keepalive RESET on" in capsys.readouterr().out


def test_no_ready_means_no_keepalive(tmp_path, monkeypatch):
    import wmagent.runner as rmod
    monkeypatch.setattr(rmod, "KEEPALIVE_S", 0.01)

    def play(ctx: PlayContext) -> None:
        time.sleep(0.05)
        ctx.game.execute("ACTION1")

    r, _ = runner(tmp_path, play, cfg=config(n_passes=1))
    result = r.run()
    assert all([s.action.name for s in g.history] == ["ACTION1"] for g in result.games)


def test_ready_failure_still_finishes_games_and_closes_the_scorecard(tmp_path, monkeypatch, capsys):
    import wmagent.runner as rmod
    monkeypatch.setattr(rmod, "KEEPALIVE_S", 10.0)
    arcade = FakeArcade(SIX, baselines=[])
    played = []

    def ready() -> None:
        raise RuntimeError("vllm failed")

    r, _ = runner(tmp_path, lambda ctx: played.append(1), cfg=config(game_ids="competition"), arcade=arcade,
                  mode=COMPETITION)
    with pytest.raises(RuntimeError, match="vllm failed"):  # offline: re-raised after teardown (test_fail_loud)
        r.run(ready)
    assert played == [] and arcade.closed == ["card-0"] and all(g.finished for g in r.games)
    assert "vllm failed" in capsys.readouterr().out
