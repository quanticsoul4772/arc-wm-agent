"""Fail fast and loud (review of the C200 runs): request failures are printed, backed off and capped; every game
records why it ended; crashes keep their traceback; a run-level error fails an offline run; the run's settings
are written next to its results; malformed settings raise; host-side failures are not passed off as engine or
model errors."""
from __future__ import annotations

import json
import sys
import time

import pytest

from tests.t1_fakes import FakeArcade
from tests.test_engine import fake_game
from tests.test_runner import SIX, config, runner
from tests.test_session import FakeClock, FakeGame, StubAgent, _session
from wmagent import engine as eng
from wmagent.engine import COMPETITION, Engine
from wmagent.runner import PlayContext, Runner
from wmagent.session import MAX_REQUEST_FAILURES, GameSession, SessionCrashed


def _bare_session(tmp_path, game, agent, **kw) -> GameSession:
    return GameSession(game=game, agent=agent, state_path=tmp_path / "s.json", transcript_path=tmp_path / "t.txt",
                       events_path=tmp_path / "e.jsonl", **kw)


def _events(tmp_path) -> list[dict]:
    return [json.loads(line) for line in (tmp_path / "e.jsonl").read_text().splitlines()]


# --------------------------------------------------------------- request failures
def test_request_failures_in_a_row_are_printed_backed_off_and_capped(tmp_path, capsys):
    sleeps: list[float] = []
    agent = StubAgent(lambda i, c: {"retryable_failure": True, "error": "request failed: ConnectionError: refused"})
    game = FakeGame()
    with pytest.raises(SessionCrashed, match="10 request failures in a row"):
        _bare_session(tmp_path, game, agent, sleep=sleeps.append).run()
    assert len(agent.calls) == MAX_REQUEST_FAILURES
    assert sleeps == [1.0, 2.0, 4.0, 8.0, 16.0, 30.0, 30.0, 30.0, 30.0]
    assert game.state == "crashed" and game.end_reason == "crash"
    out = capsys.readouterr().out
    assert out.count("[session] fake-0001 request failed") == MAX_REQUEST_FAILURES
    assert "(10/10 in a row) at analysis step 1: request failed: ConnectionError: refused" in out
    assert _events(tmp_path)[-1]["type"] == "end" and _events(tmp_path)[-1]["reason"] == "crash"


def test_a_successful_turn_resets_the_failure_count(tmp_path):
    sleeps: list[float] = []
    pattern = ([{"retryable_failure": True}] * 9 + [{"step_executed": False}]) * 2 + [None]
    agent = StubAgent(lambda i, c: pattern[i])
    with pytest.raises(SessionCrashed, match="agent returned no result"):  # not the failure cap
        _bare_session(tmp_path, FakeGame(), agent, sleep=sleeps.append).run()
    assert sleeps.count(1.0) == 2 and len(sleeps) == 18


# --------------------------------------------------------------- end reasons
def test_time_cap_end_reason_and_closing_event(tmp_path):
    clock = FakeClock()

    def script(i, call):
        clock.now += 100.0
        call["step_env"]({"action": "UP"})
        return {"step_executed": True}

    game = FakeGame()
    _bare_session(tmp_path, game, StubAgent(script), time_cap_s=250.0, clock=clock).run()
    assert game.state == "gave_up" and game.end_reason == "time_cap"
    assert game.solver_note == "tokens=0"  # unchanged format
    end = _events(tmp_path)[-1]
    assert end["type"] == "end" and end["reason"] == "time_cap" and end["elapsed_seconds"] == 300.0


@pytest.mark.parametrize(("effects", "kw", "reason"), [({}, {"max_actions": 3}, "action_cap"),
                                                       ({"ACTION1": "win"}, {}, "won")])
def test_action_cap_and_win_end_reasons(tmp_path, effects, kw, reason):
    game = FakeGame(effects=effects)
    agent = StubAgent(lambda i, c: (c["step_env"]({"action": "UP"}), {"step_executed": True})[1])
    _bare_session(tmp_path, game, agent, **kw).run()
    assert game.end_reason == reason and _events(tmp_path)[-1]["reason"] == reason


def test_stop_flag_end_reason(tmp_path):
    game = FakeGame()
    session = _bare_session(tmp_path, game, StubAgent(lambda i, c: {"step_executed": False}))
    session.agent.script = lambda i, c: (session.stop_flag.set(), {"step_executed": False})[1]
    session.run()
    assert game.end_reason == "stop_flag"


def test_finished_line_shows_end_reason_and_crash_note():
    _, game, _ = fake_game()
    game.end_reason = "time_cap"
    game.finish()
    assert game.finished_line().endswith(" end=time_cap") and game.to_json()["end_reason"] == "time_cap"
    _, crashed, _ = fake_game()
    crashed.mark_crashed("error: KeyError: 'level'")
    crashed.end_reason = "crash"
    crashed.finish()
    assert crashed.finished_line().endswith(" end=crash note=\"error: KeyError: 'level'\"")


# --------------------------------------------------------------- engine vs host failures
def test_an_engine_failure_is_an_error_payload_and_printed(tmp_path, capsys):
    session = _session(tmp_path, game=FakeGame(effects={"ACTION1": "raise"}))
    out = session.step_env({"action": "UP"})
    assert out["error"] == "UP failed in the engine: RuntimeError: engine exploded"
    assert "[session] fake-0001 engine error: UP failed in the engine" in capsys.readouterr().out


def test_a_host_failure_after_the_action_ran_is_not_called_an_engine_error(tmp_path, monkeypatch):
    session = _session(tmp_path)

    def broken(record, grid):
        raise OSError("No space left on device")

    monkeypatch.setattr(session, "_append_event", broken)
    with pytest.raises(OSError, match="No space left"):
        session.step_env({"action": "UP"})
    assert len(session.game.calls) == 1  # the action itself ran


# --------------------------------------------------------------- runner
def test_a_game_crash_prints_its_traceback(tmp_path, capsys):
    def play(ctx: PlayContext) -> None:
        if ctx.stem == "g1-00000001_p0":
            raise KeyError("level")

    r, _ = runner(tmp_path, play, cfg=config(n_passes=1))
    games = r.run().games
    crashed = [g for g in games if g.state == eng.CRASHED]
    assert len(crashed) == 1 and crashed[0].end_reason == "crash"
    captured = capsys.readouterr()
    assert "[runner] g1-00000001_p0 crashed: KeyError: 'level'" in captured.out
    assert "Traceback (most recent call last)" in captured.err and "KeyError: 'level'" in captured.err


def _failing_ready() -> None:
    raise RuntimeError("vLLM exited with code 1")


def test_a_run_level_error_fails_an_offline_run_after_writing_results(tmp_path, capsys):
    r, _ = runner(tmp_path, lambda ctx: None, cfg=config(n_passes=1))
    with pytest.raises(RuntimeError, match="vLLM exited"):
        r.run(ready=_failing_ready)
    data = json.loads((tmp_path / "benchmark.json").read_text())
    notes = {g["solver_note"] for g in data["game_runs"]}
    assert notes == {"error: RuntimeError: vLLM exited with code 1 (the run failed before this game ended)"}
    assert {g["end_reason"] for g in data["game_runs"]} == {"teardown"}
    assert "Traceback (most recent call last)" in capsys.readouterr().err


def test_a_run_level_error_in_submission_mode_still_closes_and_names_the_error(tmp_path):
    arcade = FakeArcade(SIX, baselines=[])
    cfg = config(game_ids="competition", max_runtime_s_per_game=2700, concurrency=3)
    r = Runner(cfg, lambda ctx: None, tmp_path, engine=Engine(COMPETITION, arcade=arcade), submission=True,
               notebook_start_epoch=time.time())
    result = r.run(ready=_failing_ready)  # submission mode does not raise: the scorecard must still close
    assert all(g.state == eng.CRASHED and "vLLM exited with code 1" in g.solver_note for g in result.games)


def test_run_config_is_written_without_other_env(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCAL_ANALYZER_YIELD_SECONDS", "60")
    monkeypatch.setenv("WM_MECHANISMS", "all-nocont-fbx")
    monkeypatch.setenv("KAGGLE_API_TOKEN", "not-for-the-log")
    r, _ = runner(tmp_path, lambda ctx: None, cfg=config(n_passes=1))
    r.run()
    document = json.loads((tmp_path / "run_config.json").read_text())
    assert document["env"]["LOCAL_ANALYZER_YIELD_SECONDS"] == "60"
    assert document["env"]["WM_MECHANISMS"] == "all-nocont-fbx"
    assert "not-for-the-log" not in json.dumps(document)
    assert document["run_config"]["label"] == "t1" and document["submission"] is False


# --------------------------------------------------------------- settings
def test_malformed_agent_settings_raise(monkeypatch):
    from wmagent.agent import AgentConfig

    monkeypatch.setenv("LOCAL_ANALYZER_YIELD_SECONDS", "60s")
    with pytest.raises(ValueError, match="LOCAL_ANALYZER_YIELD_SECONDS='60s' is not a valid float"):
        AgentConfig.from_env()
    monkeypatch.setenv("LOCAL_ANALYZER_YIELD_SECONDS", "60")
    monkeypatch.setenv("LOCAL_ANALYZER_ENABLE_THINKING", "ture")
    with pytest.raises(ValueError, match="LOCAL_ANALYZER_ENABLE_THINKING='ture' is not a boolean"):
        AgentConfig.from_env()
    monkeypatch.setenv("LOCAL_ANALYZER_ENABLE_THINKING", "false")
    assert AgentConfig.from_env().enable_thinking is False


def test_malformed_gate_settings_raise(monkeypatch):
    from wmagent.world_model_gate import WorldModelGate

    monkeypatch.setenv("PROBE_ENFORCE", "true")  # used to switch the gate off silently
    with pytest.raises(ValueError, match="PROBE_ENFORCE='true' must be 1 or 0"):
        WorldModelGate.from_env()
    monkeypatch.setenv("PROBE_ENFORCE", "1")
    monkeypatch.setenv("PROBE_ACTIONS", "eight")
    with pytest.raises(ValueError, match="PROBE_ACTIONS='eight' is not an integer"):
        WorldModelGate.from_env()
    monkeypatch.setenv("PROBE_ACTIONS", "8")
    monkeypatch.setenv("PROBE_ENFORCE", "0")
    assert WorldModelGate.from_env() is None


# --------------------------------------------------------------- sandbox
@pytest.mark.skipif(sys.platform == "win32", reason="the sandbox needs POSIX process groups and rlimits")
def test_a_host_handler_bug_is_printed_and_still_reported_to_the_model(capsys):
    from tests.test_sandbox import _run

    def handler(actions):
        raise AttributeError("'int' object has no attribute 'get'")

    out = _run("try:\n    action('UP')\nexcept RuntimeError as e:\n    print('model saw', e)", action_handler=handler)
    assert "model saw AttributeError" in out["stdout"]
    captured = capsys.readouterr()
    assert "[sandbox] action handler raised (reported to the model): AttributeError" in captured.out
    assert "Traceback (most recent call last)" in captured.err


def test_no_back_off_once_the_cap_is_reached(tmp_path):
    sleeps: list[float] = []
    clock = FakeClock()

    def script(i, call):
        clock.now += 300.0  # the request ran into the cap (its timeout is the time left)
        return {"retryable_failure": True, "error": "request failed: ReadTimeout"}

    game = FakeGame()
    _bare_session(tmp_path, game, StubAgent(script), time_cap_s=250.0, clock=clock, sleep=sleeps.append).run()
    assert sleeps == [] and game.end_reason == "time_cap"


@pytest.mark.skipif(sys.platform == "win32", reason="the sandbox needs POSIX process groups and rlimits")
def test_a_cell_looping_on_a_failing_handler_prints_three_times(capsys):
    from tests.test_sandbox import _run

    def handler(actions):
        raise AttributeError("boom")

    code = "for _ in range(6):\n    try:\n        action('UP')\n    except RuntimeError:\n        pass"
    _run(code, action_handler=handler)
    assert capsys.readouterr().out.count("[sandbox] action handler raised") == 3
