"""scripts/robustness_check.py: per-game shares, leave-one-game-out and end reasons, with read_arm's run checks."""
import importlib.util
from pathlib import Path

from tests.test_read_arm import _run

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "robustness_check.py"
_spec = importlib.util.spec_from_file_location("robustness_check", SCRIPT)
robustness = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(robustness)


def test_a_gain_carried_by_one_game_is_flagged(tmp_path, capsys):
    # game aa11 (even indices) gains a lot; bb22 (odd) is flat
    arm = _run(tmp_path, "arm", "x", [9.0, 1.0] * 6)
    base = _run(tmp_path, "base", "base", [0.0, 1.0] * 6)
    rc = robustness.main(["--arm", str(arm), "--arm-mechanisms", "x", "--base", str(base), "--games", "12",
                          "--draws", "2000", "--alpha", "0.05"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "aa11: arm 9.000  base 0.000  diff +9.000  share 100%" in out
    assert "without aa11: T +0.000" in out and "without bb22" in out
    assert "games whose removal loses the pass: ['aa11']" in out
    assert "end reasons arm: {'-': 12}" in out


def test_no_pass_to_lose_when_the_full_test_fails(tmp_path, capsys):
    arm = _run(tmp_path, "arm", "x", [1.0] * 12)
    base = _run(tmp_path, "base", "base", [1.0] * 12)
    assert robustness.main(["--arm", str(arm), "--arm-mechanisms", "x", "--base", str(base), "--games", "12",
                            "--draws", "500"]) == 0
    assert "no pass to lose (leave-one-out is descriptive only)" in capsys.readouterr().out


def test_run_checks_still_stop(tmp_path, capsys):
    arm = _run(tmp_path, "arm", "x", [1.0] * 12)
    base = _run(tmp_path, "base", "base", [1.0] * 12)
    assert robustness.main(["--arm", str(arm), "--arm-mechanisms", "other", "--base", str(base),
                            "--games", "12"]) == 1
    assert "RUN CHECK FAILED" in capsys.readouterr().err
