# arc-wm-agent

Clean-room implementation of an ARC-AGI-3 agent harness: a coding agent that plays games from a Python sandbox, with an enforced executable world model (`wmagent/world_model_gate.py`) and held-out confirmation. It runs offline on Kaggle with vLLM serving Qwen3.6-27B-FP8.

## Attribution and license

- **Original harness.** This code re-implements the behavior of The Duck (Tufa Labs, [`Tufalabs/duck-harness`](https://github.com/Tufalabs/duck-harness), winning solution to ARC-AGI-3 Milestone 1; [Kaggle write-up](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/717133)). It was written by agents from a behavioral spec, without access to that source; a separate verifier compared behavior. As of 2026-11-01 the Duck repository has no LICENSE file, and its `ARC3-Inference` package declares an MIT license classifier.
- **This repository** is released under CC0 1.0 (see `LICENSE`).
- **Model.** Qwen/Qwen3.6-27B and the vrfai/Qwen3.6-27B-FP8 snapshot are Apache-2.0. The optional MTP draft head (`WM_SPEC_MTP_TOKENS`) uses the BF16 `mtp.*` tensors of Qwen/Qwen3.6-27B.

## Layout

- `wmagent/engine.py`, `wmagent/runner.py`: engine adapter and runner (concurrent games, per-game cap, competition scorecard).
- `wmagent/session.py`: game session and `step_env`.
- `wmagent/sandbox.py`, `wmagent/views.py`, `wmagent/segmentation.py`: sandbox, views and segmentation.
- `wmagent/agent.py`, `wmagent/prompts.py`: agent loop and prompts.
- `wmagent/world_model_gate.py`: the world-model gate (the paper's method) and its mechanism arms (`WM_MECHANISMS`).
- `wmagent/progress.py`, `wmagent/game_class.py`: no-progress detection and game classification (Plan C mechanisms).
- `wmagent/kaggle/`: Kaggle packaging (`build.py`), vLLM setup (`setup_vllm.py`), notebook, and a local gateway simulator.
- `scripts/read_arm.py`, `scripts/robustness_check.py`: read a test arm under its pre-registered rule; per-game and leave-one-game-out checks.
- `scripts/mechanism_selftest.py`: forces every mechanism path to fire once.

## Settings (build flags → environment)

- `--wm-mechanisms` (`WM_MECHANISMS`): `base`, `m1a`, `all`, `all-nocont`, `all-nocont-fb`, `all-nocont-fbx`, `all-nocont-fbx-img`.
- `--probe-actions`, `--probe-max-refusals`, `--probe-optional`, `--probe-enforce`: world-model gate settings.
- `--spec-mtp-tokens N` (`WM_SPEC_MTP_TOKENS`): speculative decoding with the MTP head; 0 (default) is off.
- `--image-tokens N` (`LOCAL_ANALYZER_IMAGE_TOKENS`): fixed per-image cost in the context estimate; 0 (default) keeps the base64/3 estimate.

## Tests

`python -m pytest -q` (the sandbox tests need POSIX; the notebook tests need `WMAGENT_TEST_ENV_FILES` pointing at the competition's environment files).
