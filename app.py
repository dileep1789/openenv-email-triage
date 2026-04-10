from typing import Any, Dict, Optional

from fastapi import FastAPI
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
def step(payload: StepRequest) -> Dict[str, Any]:
    observation, reward, done, info = env.step(payload.action)
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
