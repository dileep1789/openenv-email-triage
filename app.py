import json
import os

import gradio as gr
from openai import OpenAI

from config_loader import get_config
from env.environment import OpenEnv
from inference import MODEL_NAME, run_episode


def run_triage(task_id: str, api_key: str):
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key

    current_key = os.getenv("OPENAI_API_KEY") or get_config("OPENAI_API_KEY")
    api_base_url = os.getenv("API_BASE_URL", "https://api.openai.com/v1")

    client = None
    status = "Mock policy used"

    if current_key and current_key != "your_key_here":
        try:
            client = OpenAI(base_url=api_base_url, api_key=current_key)
            status = f"Live model used: {MODEL_NAME}"
        except Exception as exc:
            status = f"Mock policy used due to client initialization error: {exc}"

    env = OpenEnv()
    score, trace = run_episode(env, task_id, client)
    final_state = env.state().model_dump()

    return (
        json.dumps(trace, indent=2),
        json.dumps(final_state, indent=2),
        f"Task score: {score:.4f}",
        status,
    )


with gr.Blocks(title="OpenEnv Email Ops") as demo:
    gr.Markdown("# OpenEnv Email Operations Environment")
    gr.Markdown(
        "Runs a full episode of enterprise email triage using either an OpenAI model or deterministic mock policy."
    )

    with gr.Row():
        with gr.Column():
            task_select = gr.Dropdown(
                choices=["easy_classification", "medium_response", "hard_workflow"],
                label="Select task",
                value="easy_classification",
            )
            api_input = gr.Textbox(
                label="OpenAI API key (optional)",
                placeholder="sk-...",
                type="password",
            )
            run_btn = gr.Button("Run Episode", variant="primary")
        with gr.Column():
            status_out = gr.Textbox(label="Run status")
            reward_out = gr.Textbox(label="Task score")

    with gr.Row():
        trace_out = gr.Code(label="Episode Trace (JSON)", language="json")
        state_out = gr.Code(label="Final State (JSON)", language="json")

    run_btn.click(
        fn=run_triage,
        inputs=[task_select, api_input],
        outputs=[trace_out, state_out, reward_out, status_out],
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
