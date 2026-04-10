from typing import Any, Dict, List, Tuple

from .graders import grade_phase
from .models import Action, ActionType, DecisionType, EnvState, Observation
from .tasks import EmailTask, TASKS


PHASE_NAMES = {
    1: "classification",
    2: "decision",
    3: "response",
}


class OpenEnv:
    def __init__(self):
        self.current_task: EmailTask | None = None
        self.current_state: EnvState | None = None
        self.last_feedback: str = ""

    def _allowed_actions(self, phase: int) -> List[ActionType]:
        if phase == 1:
            return [ActionType.CLASSIFY]
        if phase == 2:
            return [ActionType.DECIDE]
        return [ActionType.RESPOND]

    def _build_observation(self) -> Observation:
        if self.current_task is None or self.current_state is None:
            raise RuntimeError("Environment is not initialized. Call reset() first.")
        return Observation(
            task_id=self.current_task.id,
            difficulty=self.current_task.difficulty,
            email=self.current_task.email,
            instruction=self.current_task.instruction,
            allowed_actions=self._allowed_actions(self.current_state.phase),
            phase_name=PHASE_NAMES.get(self.current_state.phase, "response"),
            progress=min(1.0, (self.current_state.phase - 1) / 3),
            last_feedback=self.last_feedback,
        )

    def reset(self, task_name: str = "easy_classification") -> Observation:
        if task_name not in TASKS:
            raise ValueError(f"Task '{task_name}' not found. Available: {list(TASKS.keys())}")

        self.current_task = TASKS[task_name]
        self.current_state = EnvState(
            task_id=self.current_task.id,
            difficulty=self.current_task.difficulty,
            phase=1,
            max_phases=3,
            done=False,
        )
        self.last_feedback = "Episode reset. Start with email classification."
        return self._build_observation()

    def step(self, action_dict: Dict[str, Any]) -> Tuple[Observation, float, bool, Dict[str, Any]]:
        if self.current_task is None or self.current_state is None:
            raise RuntimeError("Environment is not initialized. Call reset() first.")
        if self.current_state.done:
            raise RuntimeError("Environment is finished. Call reset() to start again.")

        self.current_state.step_count += 1

        try:
            action = Action(**action_dict)
        except Exception as exc:
            self.current_state.mistakes += 1
            self.current_state.done = True
            reward = 0.0
            self.last_feedback = "Malformed action payload. Episode terminated."
            return self._build_observation(), reward, True, {"error": str(exc)}

        reward_model = grade_phase(
            phase=self.current_state.phase,
            action=action,
            task=self.current_task,
            step_count=self.current_state.step_count,
            mistakes=self.current_state.mistakes,
        )

        if reward_model.components.get("invalid_action_penalty", 0.0) < 0:
            self.current_state.mistakes += 1
        else:
            if self.current_state.phase == 1 and action.category is not None:
                self.current_state.category_identified = action.category
            if self.current_state.phase == 2 and action.decision is not None:
                self.current_state.decision_made = action.decision
            if self.current_state.phase == 3:
                self.current_state.response_sent = bool(action.response)

            if self.current_state.phase < self.current_state.max_phases:
                self.current_state.phase += 1
            else:
                self.current_state.done = True

            if (
                self.current_state.phase == 3
                and self.current_task.expected_decision == DecisionType.IGNORE
            ):
                self.current_state.done = True

        if self.current_state.step_count >= self.current_task.max_steps:
            self.current_state.done = True

        self.current_state.cumulative_reward = min(
            1.0,
            round(self.current_state.cumulative_reward + reward_model.value, 4),
        )

        for key, value in reward_model.components.items():
            self.current_state.reward_breakdown[key] = round(
                self.current_state.reward_breakdown.get(key, 0.0) + value,
                4,
            )

        self.last_feedback = reward_model.explanation
        observation = self._build_observation()
        info = {
            "reward_model": reward_model.model_dump(),
            "state": self.current_state.model_dump(),
        }
        return observation, reward_model.value, self.current_state.done, info

    def state(self) -> EnvState:
        if self.current_state is None:
            raise RuntimeError("Environment is not initialized. Call reset() first.")
        return self.current_state
