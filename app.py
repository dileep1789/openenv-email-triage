from typing import Any, Dict, Optional

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from env.environment import OpenEnv


class ResetRequest(BaseModel):
    task_id: Optional[str] = None
    task_name: Optional[str] = None


class StepRequest(BaseModel):
    action: Dict[str, Any]


app = FastAPI(title="OpenEnv Email Operations API", version="1.0.0")
env = OpenEnv()


@app.get("/")
def playground() -> HTMLResponse:
        html = """
<!doctype html>
<html>
<head>
    <meta charset=\"utf-8\" />
    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
    <title>OpenEnv Email Operations</title>
    <style>
        body { font-family: Segoe UI, Arial, sans-serif; margin: 24px; background: #f6f8fb; color: #17202a; }
        h1 { margin: 0 0 6px; }
        .muted { color: #5d6d7e; margin-bottom: 16px; }
        .card { background: #fff; border: 1px solid #d6dee8; border-radius: 10px; padding: 14px; margin-bottom: 12px; }
        button { background: #1665d8; color: #fff; border: 0; border-radius: 6px; padding: 8px 12px; cursor: pointer; }
        button.secondary { background: #3f4d5a; }
        select, textarea { width: 100%; border: 1px solid #c6d0db; border-radius: 6px; padding: 8px; }
        textarea { min-height: 120px; font-family: Consolas, monospace; }
        pre { background: #0f1720; color: #d8e0ea; padding: 12px; border-radius: 8px; overflow: auto; }
        .row { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
        @media (max-width: 850px) { .row { grid-template-columns: 1fr; } }
    </style>
</head>
<body>
    <h1>OpenEnv Email Operations</h1>
    <div class=\"muted\">Interactive API playground for reset/step/state.</div>

    <div class=\"card\">
        <label for=\"task\">Task</label>
        <select id=\"task\">
            <option value=\"easy_classification\">easy_classification</option>
            <option value=\"medium_response\">medium_response</option>
            <option value=\"hard_workflow\">hard_workflow</option>
        </select>
        <div style=\"margin-top:10px; display:flex; gap:8px;\">
            <button onclick=\"callReset()\">POST /reset</button>
            <button class=\"secondary\" onclick=\"callState()\">GET /state</button>
        </div>
    </div>

    <div class=\"row\">
        <div class=\"card\">
            <div style=\"margin-bottom:8px; font-weight:600;\">Action JSON for POST /step</div>
            <textarea id=\"action\">{
    "type": "classify",
    "category": "spam",
    "reasoning": "suspicious sender and urgency"
}</textarea>
            <div style=\"margin-top:10px;\"><button onclick=\"callStep()\">POST /step</button></div>
        </div>
        <div class=\"card\">
            <div style=\"margin-bottom:8px; font-weight:600;\">Output</div>
            <pre id=\"out\">{\"status\":\"ready\"}</pre>
        </div>
    </div>

    <script>
        const out = document.getElementById('out');
        function show(obj) { out.textContent = JSON.stringify(obj, null, 2); }
        async function callReset() {
            const task = document.getElementById('task').value;
            const r = await fetch('/reset', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ task_id: task }) });
            show(await r.json());
        }
        async function callState() {
            const r = await fetch('/state');
            show(await r.json());
        }
        async function callStep() {
            try {
                const action = JSON.parse(document.getElementById('action').value);
                const r = await fetch('/step', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ action }) });
                show(await r.json());
            } catch (e) {
                show({ error: String(e) });
            }
        }
    </script>
</body>
</html>
"""
        return HTMLResponse(content=html)


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok", "service": "openenv-email-operations"}


@app.post("/reset")
def reset(payload: Optional[ResetRequest] = None) -> Dict[str, Any]:
    task_name = "easy_classification"
    if payload is not None:
        task_name = payload.task_name or payload.task_id or task_name

    observation = env.reset(task_name)
    return observation.model_dump()


@app.post("/step")
def step(payload: Dict[str, Any]) -> Dict[str, Any]:
    action = payload.get("action", payload)
    observation, reward, done, info = env.step(action)
    return {
        "observation": observation.model_dump(),
        "reward": reward,
        "done": done,
        "info": info,
    }


@app.get("/state")
def state() -> Dict[str, Any]:
    return env.state().model_dump()


@app.post("/state")
def state_post() -> Dict[str, Any]:
    return env.state().model_dump()
