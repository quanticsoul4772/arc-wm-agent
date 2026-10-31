"""WM_MECHANISMS=all-nocont-fbx-img: all-nocont-fbx with past board images kept in the history (no M4b elision).

C2 and D generated ~40% more tokens per turn than A once their history held earlier turns: at matched history
A kept 7 board images and 13.5k characters of reasoning, C2 1 image and 19.0k characters (C201 result)."""
import json

from tests.fake_openai import FakeOpenAIServer, tool_call_response
from tests.test_agent import FakeGame, _analyze
from tests.test_plan_c_step4b import _agent
from wmagent.agent import BOARD_IMAGE_OMITTED
from wmagent.world_model_gate import ELIDE_IMAGE_MECHANISMS, FULL_MECHANISMS, MECHANISMS, WorldModelGate


def _kept_user_messages(tmp_path, arm):
    game = FakeGame(tmp_path / arm)
    script = [tool_call_response('action("UP")'), tool_call_response('action("UP")')]
    with FakeOpenAIServer(script) as server:
        agent = _agent(server, mechanisms=arm)
        _analyze(agent, game, 1)
        kept = [m for m in agent.history if m["role"] == "user"]
        _analyze(agent, game, 2)
        second = server.requests[1]["messages"]
    return kept, second


def test_kept_user_messages_keep_their_image_in_fbx_img_only(tmp_path):
    for arm, expect_image in (("all-nocont-fbx", False), ("all-nocont-fbx-img", True)):
        kept, second = _kept_user_messages(tmp_path, arm)
        has_image = any(p.get("type") == "image_url" for m in kept for p in m["content"])
        omitted = any(p.get("text") == BOARD_IMAGE_OMITTED for m in kept for p in m["content"])
        assert has_image is expect_image and omitted is not expect_image, arm
        sent = [m for m in second if m["role"] == "user"]
        assert len(sent) >= 2 and all(any(p.get("type") == "image_url" for p in m["content"]) for m in sent) \
            is expect_image, arm


def test_fbx_img_is_fbx_in_every_other_respect():
    assert "all-nocont-fbx-img" in MECHANISMS and "all-nocont-fbx-img" in FULL_MECHANISMS
    assert "all-nocont-fbx-img" not in ELIDE_IMAGE_MECHANISMS
    assert set(ELIDE_IMAGE_MECHANISMS) == set(FULL_MECHANISMS) - {"all-nocont-fbx-img"}
    for arm in ("all-nocont-fbx", "all-nocont-fbx-img"):
        gate = WorldModelGate(probe_actions=8, max_refusals=8, mechanisms=arm)
        assert gate.full and gate.capture_feedback and gate.fixes, arm
        assert gate.released_note(1) is None  # nothing released yet


def test_fbx_img_is_a_valid_build_value(tmp_path):
    from wmagent.kaggle.build import main

    main(["--out", str(tmp_path), "--label", "t", "--wm-mechanisms", "all-nocont-fbx-img", "--dry-run"])
    env = json.loads((tmp_path / "kaggle" / "source-dataset" / "agent_env.json").read_text())
    assert env["WM_MECHANISMS"] == "all-nocont-fbx-img"
