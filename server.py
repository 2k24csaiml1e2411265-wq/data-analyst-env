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
    """Reset the environment — accepts both GET and POST."""
    obs = env.reset(seed=req.seed)
    return obs


@app.post("/step", response_model=StepResult)
def step(req: StepRequest):
    """Submit an action and receive next observation + reward."""
    try:
        result = env.step(req.action)
        return result
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/state")
def state():
    """Return the current internal state of the environment."""
    return env.state()


@app.get("/tasks")
def list_tasks():
    """List all available tasks with their grading criteria."""
    from environment import TASKS
    return {"tasks": TASKS, "count": len(TASKS)}


@app.get("/health")
def health():
    return {"status": "healthy"}
