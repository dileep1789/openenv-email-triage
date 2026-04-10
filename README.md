---
title: OpenEnv Enterprise Email Operations
emoji: "📬"
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# OpenEnv Enterprise Email Operations

This repository contains a complete OpenEnv environment for a real-world task: enterprise email operations triage.

The agent must process incoming operational emails through a multi-step workflow:
1. Classify the email.
2. Choose a workflow decision (reply, escalate, ignore).
3. Draft a response when required.

The environment follows the standard OpenEnv API:
- `reset(task_id) -> Observation`
- `step(action) -> (Observation, reward, done, info)`
- `state() -> EnvState`

## Why This Is Real-World

Teams in support, trust-and-safety, and incident operations triage email continuously. The simulator captures realistic decisions:
- Distinguishing spam from operational requests.
- Selecting safe routing behavior for normal vs critical cases.
- Producing policy-safe communication for customer-facing actions.

## Environment Spec Compliance

Typed Pydantic models are implemented in `env/models.py`:
- `Observation`
- `Action`
- `Reward`
- `EnvState`
- Supporting enums (`ActionType`, `DecisionType`, `EmailCategory`)

Environment logic is implemented in `env/environment.py` and uses deterministic task definitions from `env/tasks.py` plus programmatic grading in `env/graders.py`.

Metadata and space definitions are provided in `openenv.yaml`.

## Action Space

Action payload:
```json
{
  "type": "classify | decide | respond",
  "category": "support | spam | sales | complaint | job_application",
  "decision": "reply | escalate | ignore",
  "response": "string",
  "reasoning": "string"
}
```

Phase constraints:
- `classification` phase: only `type="classify"` is valid.
- `decision` phase: only `type="decide"` is valid.
- `response` phase: only `type="respond"` is valid.

## Observation Space

Observation payload includes:
- `task_id`, `difficulty`
- `email` object (`id`, `subject`, `body`, `sender`, `history`)
- `instruction`
- `phase_name`
- `allowed_actions`
- `progress`
- `last_feedback`

## Tasks and Difficulty

Three deterministic tasks are provided:
- `easy_classification`: spam detection + safe ignore flow.
- `medium_response`: support recovery request requiring policy-safe reply.
- `hard_workflow`: enterprise outage complaint requiring escalation and incident comms.

## Reward Function

The reward is shaped across the full trajectory, not only terminal success.

Per-step signals:
- `classification` component
- `decision` component
- `response_quality` component (keyword coverage)
- `no_response_bonus` for correctly ending ignore workflows
- `safety_penalty` for unsafe response content
- `invalid_action_penalty`
- `efficiency_penalty` (loops / extra steps)

Episode score is accumulated into `EnvState.cumulative_reward` and clamped to `[0.0, 1.0]`.

## Baseline Inference (Reproducible)

`inference.py` runs all 3 tasks and prints:
- Per-phase actions and rewards
- Per-task final score
- Average score
- A machine-readable `baseline_results.json`

Reproducibility controls:
- Deterministic tasks
- `temperature=0` for model calls
- Deterministic mock policy fallback when `OPENAI_API_KEY` is missing

### Run baseline

Windows PowerShell:
```powershell
$env:OPENAI_API_KEY="your-key"
python inference.py
```

Mock-only baseline:
```powershell
python mock_inference.py
```

## Local Setup

```powershell
python -m pip install -r requirements.txt
python verify_env.py
python inference.py
```

## Hugging Face Spaces Deployment

This project includes:
- FastAPI app entrypoint: `app.py`
- Container config: `Dockerfile`

Typical HF Space setup:
1. Create a Docker Space on Hugging Face.
2. Push this repository to the Space.
3. Add secrets/env vars in Space settings:
   - `OPENAI_API_KEY` (optional)
   - `API_BASE_URL` (optional)
   - `MODEL_NAME` (optional)
4. The container starts with:
   - `uvicorn app:app --host 0.0.0.0 --port 7860`

### API Endpoints Used by Evaluator

- `POST /reset`
- `POST /step`
- `GET /state`

## OpenEnv Validation

If OpenEnv CLI is available in your runtime:
```powershell
openenv validate
```

## File Guide

- `env/models.py`: typed schema models and enums.
- `env/tasks.py`: deterministic task definitions.
- `env/graders.py`: deterministic reward/grading logic.
- `env/environment.py`: `reset/step/state` runtime logic.
- `inference.py`: OpenAI baseline runner.
- `mock_inference.py`: deterministic non-API baseline runner.
- `app.py`: Gradio app for Spaces/local demos.
- `openenv.yaml`: environment metadata/spec declaration.
