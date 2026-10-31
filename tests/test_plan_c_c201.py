"""C201: WM_MECHANISMS=all-nocont is the "all" arm without the yield-continuation prompt (M2c). Arm C slowed down
on continuation turns (1,681 generated tokens per call against 1,108 after a full prompt)."""
import json
from dataclasses import replace

import pytest

from tests.fake_openai import FakeOpenAIServer, tool_call_response
from tests.test_agent import FakeGame, FakeRefused, FakeSandbox, _analyze
from tests.test_plan_c_step2 import _user_texts, _yield_then_play
from wmagent import prompts
from wmagent.agent import CONFIG, ToolAgent
from wmagent.world_model_gate import WorldModelGate, mechanisms_from_env


def test_all_nocont_resends_the_full_prompt_after_a_yield(tmp_path):
    first, second, requests, _ = _yield_then_play(tmp_path, "all-nocont")
    assert first["yielded_control"] and second["step_executed"]
    last_user = _user_texts(requests[1])[-1]
    assert "Your turn is still open" not in last_user and last_user.startswith("Step 1, level 1.")


def test_all_nocont_keeps_every_other_all_mechanism(tmp_path):
    gate = WorldModelGate(probe_actions=1, max_refusals=8, mechanisms="all-nocont")
    assert gate.full and gate.evaluate_on_deny
    assert prompts.system_prompt("forced", 8, "all-nocont") == prompts.system_prompt("forced", 8, "all")
    game = FakeGame(tmp_path)
    script = [tool_call_response('action("UP")'), tool_call_response('action("UP")')]
    with FakeOpenAIServer(script) as server:
        config = replace(CONFIG, base_url=server.base_url, mechanisms="all-nocont")
        agent = ToolAgent(config=config, gate=gate, sandbox=(FakeSandbox(), FakeRefused))
        assert _analyze(agent, game, 1)["step_executed"]
        assert agent.progress is not None  # M2 / M4d
        kept_users = [m for m in agent.history if m["role"] == "user"]
        assert kept_users and all(isinstance(m["content"], str) or not any(
            p.get("type") == "image_url" for p in m["content"]) for m in kept_users)  # M4b
        _analyze(agent, game, 2)
    assert gate.blocked()  # M1 refusal past the 1-action probe budget


def test_all_nocont_is_a_valid_env_and_build_value(tmp_path, monkeypatch):
    from wmagent.kaggle.build import agent_env, main, parse_args

    monkeypatch.setenv("WM_MECHANISMS", "all-nocont")
    assert mechanisms_from_env() == "all-nocont"
    args = parse_args(["--out", "o", "--label", "t", "--wm-mechanisms", "all-nocont"])
    assert agent_env(args)["WM_MECHANISMS"] == "all-nocont"
    main(["--out", str(tmp_path), "--label", "t", "--wm-mechanisms", "all-nocont", "--dry-run"])
    env = json.loads((tmp_path / "kaggle" / "source-dataset" / "agent_env.json").read_text())
    assert env["WM_MECHANISMS"] == "all-nocont"
    with pytest.raises(SystemExit):
        parse_args(["--out", "o", "--label", "t", "--wm-mechanisms", "all-nocont-x"])
