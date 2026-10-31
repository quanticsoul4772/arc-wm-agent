"""Descriptive checks before submitting an arm that passed its pre-registered test (C204), stdlib only.

Not part of any ship rule. The arm is compared with a base run from another day, and the same code has scored
0.78-1.40 across runs, so a pass is shown game by game before a draw is asked about:
- the per-game mean difference (arm minus base) and how much of the total it carries;
- leave-one-game-out: the registered stratified permutation test rerun with each game dropped in turn;
- why games ended (`end_reason`, written from feat/fail-loud-telemetry on; older runs show "-");
- with `--arm` given more than once (C207: F + F2), the runs are pooled and every number above is for the pool,
  plus the harm guard as a rate (games with >= 1 level per run).

The runs are loaded and checked with scripts/read_arm.py's `load_run` (same stops: missing benchmark.json, wrong
game count, a manifest naming another arm, a missing score, a crashed game).

Usage: python -I scripts/robustness_check.py --arm RUN_DIR [--arm RUN_DIR ...] --arm-mechanisms all-nocont-fbx-img --base RUN_DIR
       [--base-mechanisms base] [--games 30] [--draws 20000] [--seed 20261029] [--alpha 0.025] [--crashes-reviewed]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

_spec = importlib.util.spec_from_file_location("read_arm", Path(__file__).resolve().parent / "read_arm.py")
read_arm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(read_arm)


GATE_SETTINGS = ("mechanisms", "probe_actions", "max_refusals", "optional", "release_after_accept")


def gate_settings(run_dir: Path) -> set[tuple]:
    """The distinct gate settings in a run's session_start manifests (one per game)."""
    out = set()
    for log in (run_dir / "artifacts").glob("*_wm_gate.jsonl"):
        start = json.loads(log.read_text(encoding="utf-8").splitlines()[0])
        out.add(tuple(start.get(k) for k in GATE_SETTINGS))
    return out


def per_game_means(runs: list[dict]) -> dict[str, float]:
    by: dict[str, list[float]] = defaultdict(list)
    for r in runs:
        by[read_arm.game_of(r)].append(float(r["final_score"]))
    return {g: statistics.mean(v) for g, v in by.items()}


def leave_one_out(arm: list[dict], base: list[dict], draws: int, seed: int) -> dict[str, tuple[float, float]]:
    """{dropped game: (T, p)} with that game's runs removed from both arms."""
    out = {}
    for game in sorted({read_arm.game_of(r) for r in arm}):
        a = [r for r in arm if read_arm.game_of(r) != game]
        b = [r for r in base if read_arm.game_of(r) != game]
        out[game] = read_arm.permutation_p(a, b, draws, seed)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--arm", type=Path, action="append", required=True, help="repeat to pool replicate runs")
    ap.add_argument("--arm-mechanisms", required=True)
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--base-mechanisms", default="base")
    ap.add_argument("--games", type=int, default=30)
    ap.add_argument("--draws", type=int, default=20_000)
    ap.add_argument("--seed", type=int, default=20261029)
    ap.add_argument("--alpha", type=float, default=0.025)
    ap.add_argument("--crashes-reviewed", action="store_true")
    a = ap.parse_args(argv)
    try:
        arm = [r for run in a.arm for r in read_arm.load_run(run, a.arm_mechanisms, a.games, a.crashes_reviewed)]
        base = read_arm.load_run(a.base, a.base_mechanisms, a.games, a.crashes_reviewed)
    except read_arm.RunCheckError as exc:
        print(f"RUN CHECK FAILED: {exc}", file=sys.stderr)
        return 1
    settings = set().union(*(gate_settings(run) for run in a.arm))
    if len(settings) != 1:  # e.g. F and H share the arm name but not the probe budget
        print(f"RUN CHECK FAILED: pooled runs differ in gate settings {GATE_SETTINGS}: {sorted(settings, key=str)}",
              file=sys.stderr)
        return 1
    for label, runs in (("arm", arm), ("base", base)):
        cleared = sum(1 for r in runs if (r.get("levels_completed") or 0) >= 1)
        print(f"{label}: {len(runs)} runs, mean {statistics.mean(float(r['final_score']) for r in runs):.3f}, "
              f"games with >= 1 level {cleared}/{len(runs)} ({cleared / len(runs):.0%})")
    ma, mb = per_game_means(arm), per_game_means(base)
    deltas = {g: ma[g] - mb.get(g, 0.0) for g in sorted(ma)}
    gain = sum(d for d in deltas.values() if d > 0)
    print("per-game mean difference (arm - base), share of the positive total:")
    for g, d in sorted(deltas.items(), key=lambda kv: -kv[1]):
        share = f"{d / gain:.0%}" if d > 0 and gain > 0 else "-"
        print(f"  {g}: arm {ma[g]:.3f}  base {mb.get(g, 0.0):.3f}  diff {d:+.3f}  share {share}")
    print(f"leave-one-game-out ({a.draws} draws, seed {a.seed}, threshold {a.alpha}):")
    loo = leave_one_out(arm, base, a.draws, a.seed)
    for g, (t, p) in loo.items():
        print(f"  without {g}: T {t:+.3f}  p {p:.4f}  {'holds' if p < a.alpha else 'does not hold'}")
    t_all, p_all = read_arm.permutation_p(arm, base, a.draws, a.seed)
    if p_all < a.alpha:
        fragile = [g for g, (_, p) in loo.items() if p >= a.alpha]
        print(f"all games: T {t_all:+.3f}  p {p_all:.4f}; games whose removal loses the pass: {fragile or 'none'}")
    else:
        print(f"all games: T {t_all:+.3f}  p {p_all:.4f}; no pass to lose (leave-one-out is descriptive only)")
    for label, runs in (("arm", arm), ("base", base)):
        print(f"end reasons {label}: {dict(Counter(r.get('end_reason') or '-' for r in runs))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
