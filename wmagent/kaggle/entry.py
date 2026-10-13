# ruff: noqa: EXE002 - the mode bit comes from the Windows bind mount, not the file
"""The notebook's runner entry point: wires the runner (T1) to the session (T2) and agent (T4).

    run(config_dict, output_dir=..., submission=..., notebook_start_epoch=...)

The agent reads its configuration from os.environ at import time (spec 2.4), so
this module imports it only when called, after setup_vllm has exported the env.
The agent builds the gate from PROBE_* (ToolAgent's default), so the session gets
no second gate.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any


def run(config: dict[str, Any], *, output_dir: str, submission: bool, notebook_start_epoch: float) -> Any:
    from wmagent.agent import ToolAgent
    from wmagent.runner import RunConfig, build_runner

    run_config = RunConfig.from_dict(config)
    prompts_dir = None if submission else Path(output_dir) / "prompts"  # submission mode: no extra artifacts
    runner = build_runner(
        run_config, output_dir, agent_factory=lambda: ToolAgent(prompts_dir=prompts_dir),
        submission=submission, notebook_start_epoch=notebook_start_epoch,
    )
    return runner.run()
