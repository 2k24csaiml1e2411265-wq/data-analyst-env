"""
Inference Script — DataAnalystEnv
===================================
MANDATORY environment variables:
  API_BASE_URL   The API endpoint for the LLM.
  MODEL_NAME     The model identifier to use for inference.
  HF_TOKEN       Your Hugging Face / API key.
"""

import os
import json
import requests
from openai import OpenAI

# ── Config ──────────────────────────────────────────────────────────────
ENV_URL    = os.environ.get("ENV_URL", "http://localhost:7860")
API_BASE   = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "")
API_KEY    = os.environ.get("HF_TOKEN") or os.environ.get("OPENAI_API_KEY", "")

client = OpenAI(base_url=API_BASE, api_key=API_KEY)

SYSTEM_PROMPT = """You are an expert data analyst. You will be given a CSV dataset and a task.
Analyze the data carefully and respond ONLY with a valid JSON object matching the task requirements.
Do not include any explanation outside the JSON object."""

TASK_PROMPTS = {
    "task_easy": """Analyze this CSV dataset and return a JSON with:
{
  "missing_values": {"column_name": count, ...},
  "data_types": {"column_name": "type_string", ...}
}""",

    "task_medium": """Analyze this CSV dataset and return a JSON with:
{
  "statistics": {
    "column_name": {"mean": float, "median": float, "std": float},
    ...
  },
  "outliers": ["col1", "col2", ...]
}""",

    "task_hard": """Analyze this CSV dataset and write a comprehensive report. Return a JSON:
{
  "report": "Your detailed analysis report covering:
  1. Data quality issues (missing values, outliers, anomalies)
  2. Key statistical insights and trends
  3. Department-level patterns
  4. Salary and performance correlations
  5. Three specific actionable business recommendations"
}"""
}

# ── Helpers ─────────────────────────────────────────────────────────────

def call_env(endpoint: str, method: str = "GET", payload: dict = None) -> dict:
    url = f"{ENV_URL}/{endpoint}"
    if method == "POST":
        r = requests.post(url, json=payload or {}, timeout=60)
    else:
        r = requests.get(url, timeout=30)
    r.raise_for_status()
    return r.json()


def agent_act(dataset_csv: str, task_id: str, task_description: str) -> dict:
    task_instruction = TASK_PROMPTS.get(task_id, task_description)
    user_message = f"""{task_instruction}

Dataset (CSV):
{dataset_csv[:6000]}"""

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_message}
        ],
        max_tokens=2000,
        temperature=0.1,
        response_format={"type": "json_object"}
    )

    raw = response.choices[0].message.content.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        print(f"  WARNING: JSON parse failed: {raw[:200]}")
        return {"raw_response": raw}


# ── Main loop ────────────────────────────────────────────────────────────

def run():
    print("=" * 60)
    print("DataAnalystEnv — Baseline Inference")
    print(f"  API_BASE_URL : {API_BASE}")
    print(f"  MODEL_NAME   : {MODEL_NAME}")
    print(f"  ENV_URL      : {ENV_URL}")
    print("=" * 60)

    print("\n[1] Resetting environment...")
    obs = call_env("reset", "POST", {"seed": 42})
    print(f"  {obs['message']}")

    total_reward = 0.0
    scores = []

    while obs.get("task") is not None:
        task = obs["task"]
        n = len(scores) + 1
        print(f"\n[Task {n}] {task['name']} ({task['difficulty'].upper()})")
        print(f"  {task['description'][:100]}...")

        print("  Calling LLM agent...")
        action = agent_act(
            dataset_csv=obs["dataset_csv"],
            task_id=task["id"],
            task_description=task["description"]
        )
        print(f"  Action keys: {list(action.keys())}")

        result = call_env("step", "POST", {"action": action})
        reward   = result["reward"]
        feedback = result["info"].get("feedback", "")
        total_reward += reward
        scores.append({"task": task["name"], "difficulty": task["difficulty"], "score": reward})

        print(f"  Score   : {reward:.3f}")
        print(f"  Feedback: {feedback}")
        obs = result["observation"]

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    for s in scores:
        bar = "█" * int(s["score"] * 20) + "░" * (20 - int(s["score"] * 20))
        print(f"  {s['difficulty'].upper():6} | {bar} | {s['score']:.3f} | {s['task']}")
    avg = total_reward / max(len(scores), 1)
    print(f"\n  Average : {avg:.3f} / 1.000")
    print(f"  Tasks   : {len(scores)}")
    print("=" * 60)
    return scores


if __name__ == "__main__":
    run()
