"""The notebook's runner entry point: wires the runner (T1) to the session (T2) and agent (T4).

    run(config_dict, output_dir=..., submission=..., notebook_start_epoch=..., ready=None)

The agent reads its configuration from os.environ at import time (spec 2.4), so
this module imports it only after setup_vllm has exported the env: at once without
`ready`, or, in a competition rerun whose server loads in the background, after
`ready()` returns (the games are made and kept alive meanwhile; see Runner.run).
The agent builds the gate from PROBE_* (ToolAgent's default), so the session gets
no second gate.
"""
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any


def run(config: dict[str, Any], *, output_dir: str, submission: bool, notebook_start_epoch: float,
        ready: Callable[[], None] | None = None) -> Any:
    from wmagent.runner import RunConfig, build_runner

    agent_cls: list[Any] = []

    def import_agent() -> None:
        from wmagent.agent import ToolAgent  # once, on this thread, after the env is exported
        agent_cls.append(ToolAgent)

    def ready_then_import() -> None:
        ready()
        import_agent()

    if ready is None:
        import_agent()
    run_config = RunConfig.from_dict(config)
    prompts_dir = None if submission else Path(output_dir) / "prompts"  # submission mode: no extra artifacts
    runner = build_runner(
        run_config, output_dir, agent_factory=lambda: agent_cls[0](prompts_dir=prompts_dir),
        submission=submission, notebook_start_epoch=notebook_start_epoch,
    )
    return runner.run(ready_then_import if ready is not None else None)
