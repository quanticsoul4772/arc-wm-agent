# arc-wm-agent

Clean-room implementation of the ARC-AGI-3 agent harness used by the paper's method: a coding agent in a Python sandbox, with an enforced executable world model (`wmagent/world_model_gate.py`) and held-out confirmation.

Built by agents from the behavioural spec in `D:\Projects\arc\claudedocs\track2-rewrite-spec-2026-10-13.md`. Builders do not read the Tufa Labs source. Only the verifier compares behaviour against it.

Layout (each module is owned by one build task):
- `wmagent/engine.py`, `wmagent/runner.py`: T1, engine adapter and runner (spec 2.1, 2.2).
- `wmagent/session.py`: T2, game session and step_env (2.3).
- `wmagent/sandbox.py`, `wmagent/views.py`, `wmagent/segmentation.py`: T3, sandbox, views and segmentation (2.5).
- `wmagent/agent.py`, `wmagent/prompts.py`: T4, agent loop and prompts (2.4).
- `wmagent/kaggle/`: T5, Kaggle packaging, vLLM setup and a local gateway simulator (2.6).
- `wmagent/world_model_gate.py`: ours, unchanged from the probe fork.
