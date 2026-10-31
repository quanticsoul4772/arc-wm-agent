"""Read a Plan C arm against its comparison arm under the pre-registered rule (C200 / C201), stdlib only.

Each run directory is a downloaded kernel output (`benchmark.json`, `artifacts/`). Before any number is computed
the run is checked, and the script stops with exit code 1 on the first problem:
- `benchmark.json` exists and holds the expected number of games;
- every game has a gate log whose `session_start` manifest names the expected WM_MECHANISMS arm;
- no game ended crashed or cancelled, and none is missing a score. The prereg keeps a crashed game's score, so
  `--crashes-reviewed` continues after a crash has been looked at; nothing else can be skipped.

Then: per-game scores, means, games with >= 1 level, the stratified one-sided permutation test (arm minus base,
scores reshuffled within each game), the harm guard, the SHIP / NO SHIP verdict, and the call-economy
secondaries from the per-request logs.

Usage: python -I scripts/read_arm.py --arm RUN_DIR --arm-mechanisms all-nocont --base RUN_DIR
       [--base-mechanisms base] [--games 30] [--draws 100000] [--seed 20261029] [--crashes-reviewed]
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path


class RunCheckError(Exception):
    pass


def load_run(run_dir: Path, mechanisms: str, games: int, crashes_reviewed: bool) -> list[dict]:
    bench = run_dir / "benchmark.json"
    if not bench.is_file():
        raise RunCheckError(f"{run_dir}: no benchmark.json (prereg: the arm is void and rerun once)")
    runs = json.loads(bench.read_text(encoding="utf-8"))["game_runs"]
    if len(runs) != games:
        raise RunCheckError(f"{run_dir}: {len(runs)} games in benchmark.json, expected {games}")
    manifests = []
    for log in sorted((run_dir / "artifacts").glob("*_wm_gate.jsonl")):
        first = log.read_text(encoding="utf-8").splitlines()[:1]
        start = json.loads(first[0]) if first else {}
        if start.get("event") != "session_start":
            raise RunCheckError(f"{log}: first record is not session_start")
        manifests.append((log.name, start.get("mechanisms")))
    wrong = [(name, arm) for name, arm in manifests if arm != mechanisms]
    if len(manifests) != games or wrong:
        raise RunCheckError(f"{run_dir}: {len(manifests)} gate manifests, expected {games} naming {mechanisms!r}; "
                            f"mismatched: {wrong[:5]}")
    missing = [r["game_id"] for r in runs if r.get("final_score") is None]
    if missing:
        raise RunCheckError(f"{run_dir}: games without a score: {missing}")
    bad = [(r["game_id"], r["state"], r.get("solver_note")) for r in runs if r["state"] in ("crashed", "cancelled")]
    if bad and not crashes_reviewed:
        raise RunCheckError(f"{run_dir}: crashed or cancelled games: {bad}. Read their notes and transcripts, then "
                            "rerun with --crashes-reviewed (the prereg keeps their recorded scores)")
    return runs


def game_of(run: dict) -> str:
    return run["game_id"].split("-")[0]


def permutation_p(arm: list[dict], base: list[dict], draws: int, seed: int) -> tuple[float, float]:
    by_game: dict[str, tuple[list[float], list[float]]] = defaultdict(lambda: ([], []))
    for r in arm:
        by_game[game_of(r)][0].append(float(r["final_score"]))
    for r in base:
        by_game[game_of(r)][1].append(float(r["final_score"]))
    n_arm, n_base = len(arm), len(base)

    def stat(groups) -> float:
        return sum(sum(a) for a, _ in groups) / n_arm - sum(sum(b) for _, b in groups) / n_base

    observed = stat(by_game.values())
    rng = random.Random(seed)
    at_least = 0
    for _ in range(draws):
        groups = []
        for a, b in by_game.values():
            pooled = a + b
            rng.shuffle(pooled)
            groups.append((pooled[:len(a)], pooled[len(a):]))
        if stat(groups) >= observed - 1e-12:
            at_least += 1
    return observed, (1 + at_least) / (1 + draws)


def call_economy(run_dir: Path) -> dict:
    """Calls per game and mean generated tokens per call, split by turn type (new step, after a yield, mid-turn);
    failed requests (rows with `error`) are counted apart."""
    kinds: dict[str, list[int]] = defaultdict(list)
    files = sorted((run_dir / "artifacts").glob("*_requests.jsonl"))
    for path in files:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        for i, row in enumerate(rows):
            if row.get("error"):
                kinds["failed"].append(0)  # a request that failed or was cut at the cap: no usage
                continue
            if row["turn_request"] != 1:
                kind = "mid_turn"
            elif i and rows[i - 1]["step"] == row["step"]:
                kind = "after_yield"
            else:
                kind = "new_step"
            kinds[kind].append(int(row.get("completion_tokens") or 0))
    calls = sum(len(v) for v in kinds.values())
    out = {"calls_per_game": round(calls / len(files), 1) if files else 0.0}
    for kind, tokens in sorted(kinds.items()):
        out[f"{kind}_calls"] = len(tokens)
        out[f"{kind}_tokens"] = round(statistics.mean(tokens)) if tokens else 0
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--arm", type=Path, required=True)
    ap.add_argument("--arm-mechanisms", required=True)
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--base-mechanisms", default="base")
    ap.add_argument("--games", type=int, default=30)
    ap.add_argument("--draws", type=int, default=100_000)
    ap.add_argument("--seed", type=int, default=20261029)
    ap.add_argument("--crashes-reviewed", action="store_true")
    a = ap.parse_args(argv)
    try:
        arm = load_run(a.arm, a.arm_mechanisms, a.games, a.crashes_reviewed)
        base = load_run(a.base, a.base_mechanisms, a.games, a.crashes_reviewed)
    except RunCheckError as exc:
        print(f"RUN CHECK FAILED: {exc}", file=sys.stderr)
        return 1
    for label, runs in (("arm", arm), ("base", base)):
        per_game: dict[str, list[float]] = defaultdict(list)
        for r in runs:
            per_game[game_of(r)].append(round(float(r["final_score"]), 2))
        print(f"{label} {'  '.join(f'{g}={v}' for g, v in sorted(per_game.items()))}")
    mean_arm = statistics.mean(float(r["final_score"]) for r in arm)
    mean_base = statistics.mean(float(r["final_score"]) for r in base)
    cleared_arm = sum(1 for r in arm if (r.get("levels_completed") or 0) >= 1)
    cleared_base = sum(1 for r in base if (r.get("levels_completed") or 0) >= 1)
    diff, p = permutation_p(arm, base, a.draws, a.seed)
    primary = p < 0.05
    guard = cleared_arm >= cleared_base
    print(f"mean arm {mean_arm:.3f}  base {mean_base:.3f}  T {diff:+.3f}  p {p:.4f} ({a.draws} draws, seed {a.seed})")
    print(f"games with >= 1 level: arm {cleared_arm}  base {cleared_base}")
    print(f"primary {'PASS' if primary else 'FAIL'}  harm guard {'PASS' if guard else 'FAIL'}  "
          f"=> {'SHIP' if primary and guard else 'NO SHIP'}")
    print(f"call economy arm  {call_economy(a.arm)}")
    print(f"call economy base {call_economy(a.base)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
