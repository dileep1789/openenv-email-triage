import json
import os
from typing import Any, Dict, List, Tuple

from openai import OpenAI

from config_loader import get_config
from env.environment import OpenEnv

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4.1-mini")
TASK_IDS = ["easy_classification", "medium_response", "hard_workflow"]


def build_prompt(observation: Dict[str, Any], current_state: Dict[str, Any]) -> str:
    return (
        "You are operating an enterprise email-triage environment. "
        "Return ONLY JSON with fields: type, category, decision, response, reasoning.\n"
        "Rules:\n"
        "- Use the exact allowed action listed in observation.allowed_actions.\n"
        "- type must be one of classify, decide, respond.\n"
        "- category is required for classify.\n"
        "- decision is required for decide and must be one of reply, escalate, ignore.\n"
        "- response should be concise and policy-safe when type is respond.\n"
        f"Observation:\n{json.dumps(observation, indent=2)}\n"
        f"State:\n{json.dumps(current_state, indent=2)}"
    )


def mock_policy(task_id: str, phase: str) -> Dict[str, Any]:
    scripted = {
        "easy_classification": {
            "classification": {
                "type": "classify",
                "category": "spam",
                "reasoning": "Promotional urgency and suspicious sender indicate spam.",
            },
            "decision": {
                "type": "decide",
                "decision": "ignore",
                "reasoning": "Spam should not receive engagement.",
            },
            "response": {
                "type": "respond",
                "response": "",
                "reasoning": "No response is appropriate for ignore workflow.",
            },
        },
        "medium_response": {
            "classification": {
                "type": "classify",
                "category": "support",
                "reasoning": "User requests account recovery assistance.",
            },
            "decision": {
                "type": "decide",
                "decision": "reply",
                "reasoning": "Support requests should be answered directly.",
            },
            "response": {
                "type": "respond",
                "response": (
                    "Thanks for contacting support. We have issued a reset link. "
                    "Please verify identity before using it. The link expires in 24 hours."
                ),
                "reasoning": "Includes security and expiry guidance.",
            },
        },
        "hard_workflow": {
            "classification": {
                "type": "classify",
                "category": "complaint",
                "reasoning": "Customer reports outage risk and contractual impact.",
            },
            "decision": {
                "type": "decide",
                "decision": "escalate",
                "reasoning": "Critical SLA incident needs escalation.",
            },
            "response": {
                "type": "respond",
                "response": (
                    "We opened an incident ticket and paged the on-call lead. "
                    "You will receive an update within 1 hour, and progress is posted on the status page."
                ),
                "reasoning": "Escalation response includes incident protocol.",
            },
        },
    }
    return scripted[task_id][phase]


def run_episode(env: OpenEnv, task_id: str, client: OpenAI | None) -> Tuple[float, List[Dict[str, Any]]]:
    observation = env.reset(task_id)
    done = False
    trace: List[Dict[str, Any]] = []

    while not done:
        state = env.state().model_dump()
        phase = observation.phase_name

        if client is None:
            action = mock_policy(task_id, phase)
        else:
            prompt = build_prompt(observation.model_dump(), state)
            response = client.chat.completions.create(
                model=MODEL_NAME,
                temperature=0,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
            )
            action = json.loads(response.choices[0].message.content)

        observation, reward, done, info = env.step(action)
        trace.append(
            {
                "phase": phase,
                "action": action,
                "reward": reward,
                "feedback": info.get("reward_model", {}).get("explanation", ""),
            }
        )

    final_score = env.state().cumulative_reward
    return final_score, trace


def main() -> None:
    api_key = os.getenv("OPENAI_API_KEY") or get_config("OPENAI_API_KEY")
    use_mock = not api_key or api_key == "your_key_here"
    client = None

    if not use_mock:
        client = OpenAI(api_key=api_key, base_url=API_BASE_URL)

    env = OpenEnv()
    scores: Dict[str, float] = {}

    print("[INFO] Running baseline inference")
    print(f"[INFO] Model mode: {'mock_policy' if client is None else MODEL_NAME}")

    for task_id in TASK_IDS:
        score, trace = run_episode(env, task_id, client)
        scores[task_id] = round(score, 4)
        print(f"\n[Task] {task_id}")
        for event in trace:
            print(f"- phase={event['phase']} reward={event['reward']:.4f} action={json.dumps(event['action'])}")
        print(f"[Task Score] {score:.4f}")

    aggregate = round(sum(scores.values()) / len(scores), 4)
    print("\n[SUMMARY]")
    print(json.dumps({"scores": scores, "average": aggregate}, indent=2))


if __name__ == "__main__":
    main()
