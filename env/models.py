from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class EmailCategory(str, Enum):
    SUPPORT = "support"
    SPAM = "spam"
    SALES = "sales"
    COMPLAINT = "complaint"
    JOB_APPLICATION = "job_application"


class DecisionType(str, Enum):
    REPLY = "reply"
    ESCALATE = "escalate"
    IGNORE = "ignore"


class ActionType(str, Enum):
    CLASSIFY = "classify"
    DECIDE = "decide"
    RESPOND = "respond"


class Email(BaseModel):
    id: str
    subject: str
    body: str
    sender: str
    history: List[str] = Field(default_factory=list)


class Action(BaseModel):
    type: ActionType
    category: Optional[EmailCategory] = None
    decision: Optional[DecisionType] = None
    response: Optional[str] = None
    reasoning: Optional[str] = None


class Reward(BaseModel):
    value: float = Field(ge=0.0, le=1.0)
    components: Dict[str, float] = Field(default_factory=dict)
    progress: float = Field(ge=0.0, le=1.0)
    explanation: str


class EnvState(BaseModel):
    task_id: str
    difficulty: str
    phase: int
    max_phases: int
    done: bool
    category_identified: Optional[EmailCategory] = None
    decision_made: Optional[DecisionType] = None
    response_sent: bool = False
    step_count: int = 0
    cumulative_reward: float = Field(default=0.0, ge=0.0, le=1.0)
    reward_breakdown: Dict[str, float] = Field(default_factory=dict)
    mistakes: int = 0


class Observation(BaseModel):
    task_id: str
    difficulty: str
    email: Email
    instruction: str
    allowed_actions: List[ActionType]
    phase_name: str
    progress: float = Field(ge=0.0, le=1.0)
    last_feedback: str = ""
