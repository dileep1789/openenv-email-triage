import json
from env.environment import OpenEnv

# Environment setup
env = OpenEnv()

print("[START - MOCK MODE]")

tasks = ["easy_classification", "medium_response", "hard_workflow"]

# Mock responses for the environment
MOCK_RESPONSES = {
    "easy_classification": {
        "type": "classify",
        "category": "spam",
        "reasoning": "This is clearly a promotional/spam email about a free iPhone."
    },
    "medium_response": {
        "type": "reply",
        "category": "support",
        "response": "Hello, thank you for contacting support. To help you with your password reset request, I've sent a secure reset link to your registered email address. Please follow the security instructions to regain access. Let us know if you need further assistance.",
        "reasoning": "User needs a password reset, so categorization is support and a reply is required."
    },
    "hard_workflow": {
        "type": "escalate",
        "category": "job_application",
        "response": "Forwarding this application to the recruiting team and hiring manager for review.",
        "reasoning": "As this is a professional job application for a Senior Developer, I will escalate this to HR and the engineering hiring manager."
    }
}

for task_id in tasks:
    print(f"\n--- Running Stage: {task_id.replace('_', ' ').title()} ---")
    
    # Initialize environment
    observation = env.reset(task_id)
    print(f"Observation ID: {observation.email.id}")
    print(f"Subject: {observation.email.subject}")
    print(f"Instruction: {observation.instruction}")
    
    # Simulate LLM action
    action_dict = MOCK_RESPONSES.get(task_id)
    
    # Take step
    next_obs, reward, done, info = env.step(action_dict)
    
    print(f"[ACTION] Type: {action_dict['type']}, Category: {action_dict['category']}")
    print(f"[REWARD] {reward}")
    print(f"[DONE]   {done}")

print("\n[END - MOCK RUN COMPLETED]")
