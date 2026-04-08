"""
FastAPI server for EmailTriageEnv OpenEnv compatibility.
Exposes typical OpenEnv endpoints using the Pydantic models.
"""

import os
from typing import Any, Dict

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from email_triage_env.environment import EmailTriageEnv
from email_triage_env.models import EmailAction, EmailObservation


app = FastAPI(title="EmailTriageEnv OpenEnv API", version="1.0.0")

class ResetRequest(BaseModel):
    task_id: str = "basic_triage"
    seed: int = 42

class StepResponse(BaseModel):
    observation: EmailObservation
    reward: float
    done: bool
    info: Dict[str, Any]

class EnvironmentState:
    env: EmailTriageEnv | None = None

state = EnvironmentState()

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/reset", response_model=EmailObservation)
def reset(req: ResetRequest | None = None):
    if req is None:
        req = ResetRequest()
    state.env = EmailTriageEnv(task_id=req.task_id, seed=req.seed)
    obs = state.env.reset()
    return obs

@app.post("/step", response_model=StepResponse)
def step(action: EmailAction):
    if state.env is None:
        raise HTTPException(status_code=400, detail="Environment not initialized via /reset")
    obs, reward, done, info = state.env.step(action)
    return StepResponse(
        observation=obs,
        reward=reward,
        done=done,
        info=info
    )

@app.get("/state")
def get_state():
    if state.env is None:
        raise HTTPException(status_code=400, detail="Environment not initialized via /reset")
    return state.env.state()

@app.post("/grade")
def grade():
    if state.env is None:
        raise HTTPException(status_code=400, detail="Environment not initialized via /reset")
    return {"score": state.env.grade()}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)
