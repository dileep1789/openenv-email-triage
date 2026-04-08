import gradio as gr
import json
import os
from openai import OpenAI
from env.environment import OpenEnv
from config_loader import get_config

def run_triage(task_id, api_key):
    """
    Runs the triage logic for a specific task using the provided API key.
    """
    # Override environment if API key is provided manually in UI
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
    
    # Configuration - Matching submission requirement pattern
    API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
    MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4-turbo-preview")
    HF_TOKEN = os.getenv("HF_TOKEN")
    LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")
    
    current_key = os.getenv("OPENAI_API_KEY")
    if not current_key:
        from config_loader import get_config
        current_key = get_config("OPENAI_API_KEY")
    
    env = OpenEnv()
    
    try:
        if not current_key or current_key == "your_key_here":
            use_mock = True
        else:
            client = OpenAI(base_url=API_BASE_URL, api_key=current_key)
            use_mock = False
    except Exception:
        use_mock = True

    # Reset environment
    observation = env.reset(task_id)
    state_json = json.dumps(observation.model_dump(), indent=2)
    
    if use_mock:
        # Predefined mock responses
        MOCK_RESPONSES = {
            "easy_classification": {"type": "classify", "category": "spam", "reasoning": "Mocking spam detection."},
            "medium_response": {"type": "reply", "category": "support", "response": "Hello, thank you for contacting support!", "reasoning": "Mocking support reply."},
            "hard_workflow": {"type": "escalate", "category": "job_application", "response": "Forwarded to HR.", "reasoning": "Mocking escalation."}
        }
        action_dict = MOCK_RESPONSES.get(task_id, {"type": "ignore", "category": "spam"})
        status = "Used Mock Agent (No API Key found)"
    else:
        prompt = f"""
        You are an AI email triage assistant. Based on the following observation, decide on an action.
        Observation: {state_json}
        Return ONLY valid JSON with keys: type, category, response, reasoning.
        """
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        action_dict = json.loads(response.choices[0].message.content)
        status = f"Success using {MODEL_NAME}"

    # Take step
    next_obs, reward, done, info = env.step(action_dict)
    
    return (
        state_json, 
        json.dumps(action_dict, indent=2), 
        f"Reward: {reward}", 
        status
    )

# UI Layout
with gr.Blocks(title="OpenEnv Email Triage") as demo:
    gr.Markdown("# 📧 OpenEnv Email Triage AI")
    gr.Markdown("Test your AI agent in a simulated email environment.")
    
    with gr.Row():
        with gr.Column():
            task_select = gr.Dropdown(
                choices=["easy_classification", "medium_response", "hard_workflow"],
                label="Select Task",
                value="easy_classification"
            )
            api_input = gr.Textbox(
                label="OpenAI API Key (Optional, overrides token.txt)",
                placeholder="sk-...",
                type="password"
            )
            run_btn = gr.Button("Run Agent", variant="primary")
            
        with gr.Column():
            status_out = gr.Label(label="Status")
            reward_out = gr.Textbox(label="Result")

    with gr.Row():
        obs_out = gr.Code(label="Input Observation (JSON)", language="json")
        act_out = gr.Code(label="Agent Action (JSON)", language="json")

    run_btn.click(
        fn=run_triage,
        inputs=[task_select, api_input],
        outputs=[obs_out, act_out, reward_out, status_out]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
