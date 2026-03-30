import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from models import Observation, StepResult, ResetRequest, StepRequest
from environment import DataAnalystEnv

app = FastAPI(
    title="DataAnalystEnv",
    description="OpenEnv environment — AI agent learns to analyze real-world employee datasets",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

env = DataAnalystEnv()


@app.get("/")
def root():
    return {"status": "ok", "env": "DataAnalystEnv", "version": "1.0.0"}


@app.api_route("/reset", methods=["GET", "POST"], response_model=Observation)
def reset(req: ResetRequest = ResetRequest()):
    obs = env.reset(seed=req.seed)
    return obs


@app.post("/step", response_model=StepResult)
def step(req: StepRequest):
    try:
        result = env.step(req.action)
        return result
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/state")
def state():
    return env.state()


@app.get("/tasks")
def list_tasks():
    from environment import TASKS
    return {"tasks": TASKS, "count": len(TASKS)}


@app.get("/health")
def health():
    return {"status": "healthy"}
