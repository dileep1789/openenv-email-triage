from typing import Dict, List

from pydantic import BaseModel, Field

from .models import DecisionType, Email, EmailCategory


class EmailTask(BaseModel):
    id: str
    difficulty: str
    email: Email
    expected_category: EmailCategory
    expected_decision: DecisionType
    required_reply_keywords: List[str] = Field(default_factory=list)
    instruction: str
    max_steps: int = 5


TASKS: Dict[str, EmailTask] = {
    "easy_classification": EmailTask(
        id="easy_classification",
        difficulty="easy",
        email=Email(
            id="e-001",
            subject="Last chance: claim your gift card today",
            body=(
                "Congratulations! Click this external link to claim a free gift card. "
                "This offer expires in 30 minutes."
            ),
            sender="promo-alerts@unknown-prize.biz",
            history=["No prior conversation."],
        ),
        expected_category=EmailCategory.SPAM,
        expected_decision=DecisionType.IGNORE,
        instruction=(
            "You are handling incoming inbox triage. First classify the email. "
            "Then decide whether to reply, escalate, or ignore."
        ),
        max_steps=4,
    ),
    "medium_response": EmailTask(
        id="medium_response",
        difficulty="medium",
        email=Email(
            id="m-001",
            subject="Cannot access my account after MFA reset",
            body=(
                "Hi support, after my phone reset I cannot log in. "
                "Can you help me recover access to my account?"
            ),
            sender="employee@acme-corp.com",
            history=["User reported issue 2 hours ago via chatbot."],
        ),
        expected_category=EmailCategory.SUPPORT,
        expected_decision=DecisionType.REPLY,
        required_reply_keywords=["reset link", "verify identity", "24 hours"],
        instruction=(
            "Classify the request, decide the workflow action, and send a policy-safe reply."
        ),
    ),
    "hard_workflow": EmailTask(
        id="hard_workflow",
        difficulty="hard",
        email=Email(
            id="h-001",
            subject="Enterprise outage complaint - contract penalty risk",
            body=(
                "Our payment API has been failing for 47 minutes. "
                "This could trigger contractual penalties. We need immediate senior support."
            ),
            sender="ops-director@enterprise-client.com",
            history=[
                "Priority customer with platinum SLA.",
                "Previous critical incident was handled by incident response lead.",
            ],
        ),
        expected_category=EmailCategory.COMPLAINT,
        expected_decision=DecisionType.ESCALATE,
        required_reply_keywords=["incident ticket", "on-call", "within 1 hour", "status page"],
        instruction=(
            "Process this high-risk customer email. Classify it, choose the safest operational action, "
            "and provide a concise escalation message."
        ),
    ),
}
