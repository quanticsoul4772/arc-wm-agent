"""WM_SPEC_MTP_TOKENS / --spec-mtp-tokens: speculative decoding with the grafted BF16 MTP head (paths.MTP_REF).

The FP8 snapshot ships `mtp_num_hidden_layers: 1` but no MTP weights, so vLLM would draft with a random head; the
graft adds them, and a startup check fails loud unless the mean acceptance length shows the head works."""
import io
import json

import pytest

from wmagent.kaggle import paths, setup_vllm


def test_spec_tokens_parse_strictly():
    assert setup_vllm.spec_tokens({}) == 0
    assert setup_vllm.spec_tokens({"WM_SPEC_MTP_TOKENS": "2"}) == 2
    for bad in ("two", "-1"):
        with pytest.raises(ValueError, match="WM_SPEC_MTP_TOKENS"):
            setup_vllm.spec_tokens({"WM_SPEC_MTP_TOKENS": bad})


def test_vllm_command_adds_the_speculative_config_only_when_on(tmp_path):
    assert "--speculative-config" not in setup_vllm.vllm_command(tmp_path)
    cmd = setup_vllm.vllm_command(tmp_path, spec=2)
    assert json.loads(cmd[cmd.index("--speculative-config") + 1]) == {"method": "mtp", "num_speculative_tokens": 2}


def test_the_head_is_grafted_and_kept_out_of_fp8(tmp_path):
    staged, head = tmp_path / "model", tmp_path / "mtp"
    staged.mkdir()
    head.mkdir()
    (staged / "config.json").write_text(json.dumps({"quantization_config": {"ignore": ["lm_head"]}, "x": 1}))
    (staged / "model.safetensors").write_bytes(b"w")
    (head / "mtp.safetensors").write_bytes(b"mtp")
    setup_vllm.add_mtp_head(staged, head)
    setup_vllm.add_mtp_head(staged, head)  # idempotent
    config = json.loads((staged / "config.json").read_text())
    assert config["quantization_config"]["ignore"] == ["lm_head", "re:.*mtp.*"] and config["x"] == 1
    assert (staged / "mtp.safetensors").read_bytes() == b"mtp"
    assert paths.is_mtp_dir(head) and not paths.is_mtp_dir(staged) and not paths.is_model_dir(head)


class _Reply(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _check(tmp_path, monkeypatch, logged):
    log = tmp_path / "server.log"
    log.write_text("startup\n")

    def fake_urlopen(req, timeout):
        body = json.loads(req.data)
        assert body["max_tokens"] >= 200 and body["temperature"] == 0.0
        if logged is not None:
            with open(log, "a") as f:
                f.write(f"INFO SpecDecoding metrics: Mean acceptance length: {logged}, Accepted: 10 tokens\n")
        return _Reply(b'{"choices": [{"message": {"content": "1, 2, 3"}}]}')

    monkeypatch.setattr(setup_vllm.urllib.request, "urlopen", fake_urlopen)
    return setup_vllm.check_spec_acceptance("http://x/v1", log, timeout_s=0.5)


def test_acceptance_check_passes_a_working_head(tmp_path, monkeypatch):
    assert _check(tmp_path, monkeypatch, "2.41") == 2.41


def test_acceptance_check_fails_a_head_that_did_not_load(tmp_path, monkeypatch):
    with pytest.raises(RuntimeError, match="mean acceptance length 1.02 < 1.3"):
        _check(tmp_path, monkeypatch, "1.02")


def test_acceptance_check_fails_without_metrics(tmp_path, monkeypatch):
    with pytest.raises(RuntimeError, match="no 'Mean acceptance length' line"):
        _check(tmp_path, monkeypatch, None)


def test_spec_without_the_head_dataset_fails_before_anything_starts(tmp_path):
    with pytest.raises(ValueError, match="needs the MTP head dataset"):
        setup_vllm.start(model_dir=tmp_path, wheelhouse=tmp_path, working=tmp_path,
                         agent_env={"WM_SPEC_MTP_TOKENS": "2"}, check_gpu_first=False)


def test_build_attaches_the_head_and_the_setting(tmp_path):
    from wmagent.kaggle.build import main

    main(["--out", str(tmp_path / "on"), "--label", "t", "--wm-mechanisms", "base", "--spec-mtp-tokens", "2",
          "--dry-run"])
    main(["--out", str(tmp_path / "off"), "--label", "t", "--wm-mechanisms", "base", "--dry-run"])
    for name, sources, value in (("on", True, "2"), ("off", False, "0")):
        meta = json.loads((tmp_path / name / "kaggle" / "kernel" / "kernel-metadata.json").read_text())
        env = json.loads((tmp_path / name / "kaggle" / "source-dataset" / "agent_env.json").read_text())
        assert (paths.MTP_REF in meta["dataset_sources"]) is sources and env["WM_SPEC_MTP_TOKENS"] == value
