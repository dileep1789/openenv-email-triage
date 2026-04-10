from .models import Action, ActionType, DecisionType, Reward
from .tasks import EmailTask


def _keyword_match_fraction(text: str, keywords: list[str]) -> float:
    if not keywords:
        return 1.0
    if not text:
        return 0.0
    text_lower = text.lower()
    hits = sum(1 for keyword in keywords if keyword.lower() in text_lower)
    return hits / len(keywords)


def grade_phase(
    *,
    phase: int,
    action: Action,
    task: EmailTask,
    step_count: int,
    mistakes: int,
) -> Reward:
    components: dict[str, float] = {
        "classification": 0.0,
        "decision": 0.0,
        "response_quality": 0.0,
        "efficiency_penalty": 0.0,
        "invalid_action_penalty": 0.0,
    }
    explanation = ""

    expected_type = {
        1: ActionType.CLASSIFY,
        2: ActionType.DECIDE,
        3: ActionType.RESPOND,
    }.get(phase, ActionType.RESPOND)

    if action.type != expected_type:
        components["invalid_action_penalty"] = -0.08
        explanation = f"Expected action type '{expected_type.value}', received '{action.type.value}'."
    elif phase == 1:
        if action.category == task.expected_category:
            components["classification"] = 0.35
            explanation = "Correct email category identified."
        else:
            components["classification"] = 0.05
            explanation = "Category selected but incorrect."
    elif phase == 2:
        if action.decision == task.expected_decision:
            components["decision"] = 0.35
            explanation = "Correct operational decision chosen."
        else:
            safe_partial = {
                DecisionType.ESCALATE: 0.10,
                DecisionType.REPLY: 0.05,
                DecisionType.IGNORE: 0.0,
            }
            components["decision"] = safe_partial.get(action.decision, 0.0)
            explanation = "Decision made, but it does not fully match task expectations."
    else:
        if task.expected_decision == DecisionType.IGNORE:
            if action.response:
                components["response_quality"] = 0.0
                explanation = "No response was needed for ignore workflow."
            else:
                components["response_quality"] = 0.30
                explanation = "Correctly avoided unnecessary response."
        else:
            quality = _keyword_match_fraction(action.response or "", task.required_reply_keywords)
            components["response_quality"] = round(0.30 * quality, 4)
            explanation = "Response scored by required policy keyword coverage."

    if step_count > 3:
        components["efficiency_penalty"] = -0.03 * (step_count - 3)
    if mistakes > 0:
        components["efficiency_penalty"] -= 0.02 * mistakes

    phase_reward = sum(components.values())
    phase_reward = max(0.0, min(1.0, round(phase_reward, 4)))
    progress = min(1.0, phase / 3)

    return Reward(
        value=phase_reward,
        components=components,
        progress=progress,
        explanation=explanation,
    )
