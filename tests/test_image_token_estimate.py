"""LOCAL_ANALYZER_IMAGE_TOKENS: the context estimate charges a fixed cost per image part instead of its base64 text/3.

At 0 (the default, every arm up to C204) nothing changes. A board's ~1,440 base64 characters were estimated at
~480 tokens where the model was charged ~170, so arms that keep images trimmed their text history early."""
import json

from wmagent.agent import TOOL_CHOICE, TOOLS, estimate_tokens, persistable_history, trim_messages

IMAGE = {"type": "image_url", "image_url": {"url": "data:image/png;base64," + "A" * 1440}}


def _user(text, image=True):
    return {"role": "user", "content": [{"type": "text", "text": text}, *([IMAGE] if image else [])]}


def _conversation(turns):
    messages = [{"role": "system", "content": "s" * 300}]
    for i in range(turns):
        messages += [_user(f"turn {i} " + "x" * 600), {"role": "assistant", "content": "y" * 900}]
    return messages


def test_default_estimate_is_unchanged():
    messages = _conversation(3)
    payload = {"messages": messages, "tools": TOOLS, "tool_choice": TOOL_CHOICE}
    assert estimate_tokens(messages) == len(json.dumps(payload, sort_keys=True, ensure_ascii=True)) / 3
    assert estimate_tokens(messages, 0) == estimate_tokens(messages)


def test_fixed_image_cost_replaces_the_base64_length():
    with_images, without = _conversation(3), [m if m["role"] != "user" else _user(m["content"][0]["text"], False)
                                              for m in _conversation(3)]
    base64_cost = estimate_tokens(with_images) - estimate_tokens(without)
    fixed_cost = estimate_tokens(with_images, 170) - estimate_tokens(without, 170)
    assert base64_cost > 3 * 470  # ~480 per image
    assert 3 * 170 <= fixed_cost < 3 * 190  # 170 each plus the emptied part's few characters
    assert with_images[1]["content"][1]["image_url"]["url"].startswith("data:image/png")  # not mutated


def test_a_fixed_image_cost_keeps_more_history_under_the_same_budget():
    messages = _conversation(12)
    budget = estimate_tokens(messages, 170) * 0.8
    legacy, fixed = trim_messages(messages, budget), trim_messages(messages, budget, 170)
    assert len(fixed) > len(legacy)
    assert len(persistable_history(messages, budget, 170)) > len(persistable_history(messages, budget))


def test_setting_reaches_the_agent_config_and_the_build(monkeypatch, tmp_path):
    from wmagent.agent import AgentConfig
    from wmagent.kaggle.build import main

    monkeypatch.setenv("LOCAL_ANALYZER_IMAGE_TOKENS", "170")
    assert AgentConfig.from_env().image_tokens == 170
    monkeypatch.delenv("LOCAL_ANALYZER_IMAGE_TOKENS")
    assert AgentConfig.from_env().image_tokens == 0
    main(["--out", str(tmp_path), "--label", "t", "--wm-mechanisms", "base", "--image-tokens", "170", "--dry-run"])
    env = json.loads((tmp_path / "kaggle" / "source-dataset" / "agent_env.json").read_text())
    assert env["LOCAL_ANALYZER_IMAGE_TOKENS"] == "170"
