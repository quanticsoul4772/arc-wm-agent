"""Plan C step 3 (M3): one classifier and one gate-profile source for the gate, the prompt and the logs."""
import json

import pytest

from wmagent import prompts
from wmagent.game_class import CLASS_OVERRIDES, GAME_CLASSES, classify, profile_for
from wmagent.world_model_gate import CHANGE_FLOOR, MIN_INFORMATIVE, PASS_ACCURACY, WorldModelGate


@pytest.mark.parametrize(("labels", "expected"), [
    (["MOUSE"], "click"),
    (["ACTION6", "RESET"], "click"),
    (["UP", "DOWN", "LEFT", "RIGHT"], "keys"),
    (["SPACE", "ACTION7"], "keys"),
    (["UP", "MOUSE"], "mixed"),
    (["RESET"], None),
    ([], None),
])
def test_classify(labels, expected):
    assert classify(labels) == expected


def test_enforced_values_are_todays_constants_for_every_class():
    gate = WorldModelGate(probe_actions=8, max_refusals=8)
    base = gate.base_profile
    assert (base.change_floor, base.min_informative, base.pass_accuracy) == (CHANGE_FLOOR, MIN_INFORMATIVE,
                                                                            PASS_ACCURACY)
    assert all(CLASS_OVERRIDES[c] == {} for c in GAME_CLASSES)
    assert all(profile_for(c, base) == base for c in GAME_CLASSES)


def test_prompt_numbers_equal_the_profile_numbers_for_every_class():
    for game_class in GAME_CLASSES:
        gate = WorldModelGate(probe_actions=8, max_refusals=8, mechanisms="m1a", game_class=game_class)
        p = gate.profile
        text = prompts.system_prompt("forced", 8, "m1a", p)
        assert f"may spend {p.probe_actions} real actions" in text
        assert f"changed more than {p.change_floor} cells" in text
        assert f"at least {p.min_informative} such moves" in text
        assert f"at least {int(p.pass_accuracy * 100)}% of them" in text


def test_gate_tags_every_record_with_the_class(tmp_path):
    gate = WorldModelGate(probe_actions=1, max_refusals=8)
    gate.reset(tmp_path / "g_p0_tool_runtime_state.json")
    gate.observe_level(1)
    gate.observe_class(["UP", "DOWN"])
    gate.observe_class(["UP", "DOWN"])
    gate.observe_class([])  # terminal: keeps the class
    gate.record_actions(1)
    assert gate.check_action(1) is not None
    records = [json.loads(x) for x in gate.log_path.read_text().splitlines()]
    assert [r["event"] for r in records] == ["session_start", "refused"]
    assert records[0]["game_class"] is None and records[1]["game_class"] == "keys"
    gate.observe_class(["MOUSE"])
    gate.check_action(1)
    assert json.loads(gate.log_path.read_text().splitlines()[-1])["game_class"] == "click"


def test_refusals_and_caps_record_what_the_shadow_profiles_would_do(tmp_path):
    gate = WorldModelGate(probe_actions=2, max_refusals=8)
    gate.reset(tmp_path / "g_p0_tool_runtime_state.json")
    gate.observe_level(1)
    gate.observe_class(["MOUSE"])
    assert gate.check_action(3) is not None  # capped: 3 > 2 remaining
    gate.record_actions(2)
    assert gate.check_action(1) is not None  # refused at 2 actions
    records = [json.loads(x) for x in gate.log_path.read_text().splitlines()]
    capped = next(r for r in records if r["event"] == "batch_capped")
    refused = next(r for r in records if r["event"] == "refused")
    assert capped["shadow"] == {"optional": False, "probe16": False}
    assert refused["shadow"] == {"optional": False, "probe16": False}
    gate.record_actions(14)
    gate.check_action(1)
    last = json.loads(gate.log_path.read_text().splitlines()[-1])
    assert last["shadow"] == {"optional": False, "probe16": True}


def test_agent_classifies_from_the_turns_labels(tmp_path):
    from tests.fake_openai import FakeOpenAIServer, error_response, tool_call_response
    from tests.test_agent import FakeGame, _agent, _analyze

    game = FakeGame(tmp_path)
    gate = WorldModelGate(probe_actions=8, max_refusals=8)
    with FakeOpenAIServer([tool_call_response('action("UP")'), error_response(503, "end")]) as server:
        _analyze(_agent(server, gate=gate), game)
    assert gate.game_class == "keys"
