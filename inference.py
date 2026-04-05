import os
import json
from openai import OpenAI
from env.environment import OpenEnv

# Initialize OpenAI client (or fallback to Mock Agent)
try:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY missing. Falling back to Mock Agent.")
    
    client = OpenAI(
        base_url=os.getenv("API_BASE_URL", "https://api.openai.com/v1"),
        api_key=api_key
    )
    USE_MOCK = False
except Exception as e:
    print(f"[INFO] {e}")
    USE_MOCK = True

model_name = os.getenv("MODEL_NAME", "gpt-4-turbo-preview")

# Predefined mock responses for local testing
MOCK_RESPONSES = {
    "easy_classification": {"type": "classify", "category": "spam", "reasoning": "Mocking spam detection."},
    "medium_response": {"type": "reply", "category": "support", "response": "Hello, thank you for contacting support! We've sent you a password reset link.", "reasoning": "Mocking support reply."},
    "hard_workflow": {"type": "escalate", "category": "job_application", "response": "Forwarded to HR.", "reasoning": "Mocking escalation."}
}

print("[START]")

tasks = ["easy_classification", "medium_response", "hard_workflow"]

# Environment setup (must be here to avoid scope issue)
env = OpenEnv()

for task_id in tasks:
    print(f"--- Task: {task_id} ---")
    
    # Initialize environment for the task
    observation = env.reset(task_id)
    
    if USE_MOCK:
        action_dict = MOCK_RESPONSES.get(task_id)
    else:
        state_json = json.dumps(observation.model_dump(), indent=2)
        
        # Prompting the LLM
        prompt = f"""
        You are an AI email triage assistant. Based on the following observation, decide on an action.
        
        Observation:
        {state_json}
        
        Your action MUST be a JSON object with:
        - type: One of [classify, reply, escalate, ignore]
        - category: One of [support, spam, sales, complaint, job_application]
        - response: String (optional, required if type is 'reply' or 'escalate')
        - reasoning: String (optional, explain your decision)
        
        Return ONLY valid JSON.
        """
        
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        
        action_dict = json.loads(response.choices[0].message.content)
    
    # Take step
    next_obs, reward, done, info = env.step(action_dict)
    
    print(f"[ACTION] {json.dumps(action_dict)}")
    print(f"[REWARD] {reward}")

print("[END]")
