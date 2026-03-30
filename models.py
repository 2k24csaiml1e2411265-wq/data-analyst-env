from pydantic import BaseModel
from typing import Optional, Any


class TaskInfo(BaseModel):
    id: str
    name: str
    difficulty: str  # easy | medium | hard
    description: str


class Observation(BaseModel):
    dataset_csv: Optional[str] = None
    task: Optional[TaskInfo] = None
    message: str = ""


class StepResult(BaseModel):
    observation: Observation
    reward: float          # 0.0 – 1.0
    done: bool
    info: dict[str, Any] = {}


class ResetRequest(BaseModel):
    seed: Optional[int] = None


class StepRequest(BaseModel):
    action: dict[str, Any]
