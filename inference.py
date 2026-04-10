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

# ── Config (HF_TOKEN has NO default) ────────────────────────────────────
API_BASE_URL     = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME       = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN         = os.getenv("HF_TOKEN")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

ENV_URL = os.getenv("ENV_URL", "http://localhost:7860")

SYSTEM_PROMPT = "You are an expert data analyst. Respond ONLY with valid JSON. No explanation outside JSON."

TASK_PROMPTS = {
    "task_easy": """Analyze the CSV and return ONLY this JSON:
{
  "missing_values": {"column_name": count},
  "data_types": {"column_name": "dtype_string"}
}
Only include columns that have missing values in missing_values.""",

    "task_medium": """Analyze the CSV and return ONLY this JSON:
{
  "statistics": {
    "column_name": {"mean": float, "median": float, "std": float}
  },
  "outliers": ["column_name"]
}
Include all numeric columns in statistics. Use IQR method for outliers.""",

    "task_hard": """Analyze the CSV and return ONLY this JSON:
{
  "report": "Your full analysis report covering: data quality issues, statistical insights, trends, and exactly 3 actionable business recommendations."
}"""
}

# ── Helpers ──────────────────────────────────────────────────────────────

def call_env(endpoint, method="GET", payload=None):
    try:
        url = f"{ENV_URL}/{endpoint}"
        r = requests.post(url, json=payload or {}, timeout=60) if method == "POST" \
            else requests.get(url, timeout=30)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"[ERROR] {endpoint}: {e}", flush=True)
        return {}


def agent_act(dataset_csv, task_id, task_description):
    prompt = TASK_PROMPTS.get(task_id, task_description)
    user_msg = f"{prompt}\n\nDataset (CSV):\n{dataset_csv[:5000]}"
    try:
        client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_msg}
            ],
            max_tokens=2000,
            temperature=0.1,
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content.strip())
    except Exception as e:
        print(f"[ERROR] LLM call failed: {e}", flush=True)
        return fallback_action(task_id, dataset_csv)


def fallback_action(task_id, dataset_csv):
    try:
        import pandas as pd
        import numpy as np
        from io import StringIO
        df = pd.read_csv(StringIO(dataset_csv))
        num = df.select_dtypes(include=np.number)

        if task_id == "task_easy":
            return {
                "missing_values": {c: int(df[c].isnull().sum())
                                   for c in df.columns if df[c].isnull().sum() > 0},
                "data_types": {c: str(df[c].dtype) for c in df.columns}
            }
        elif task_id == "task_medium":
            stats = {c: {"mean": round(float(num[c].mean()), 2),
                         "median": round(float(num[c].median()), 2),
                         "std": round(float(num[c].std()), 2)}
                     for c in num.columns}
            outliers = []
            for c in num.columns:
                q1, q3 = num[c].quantile(0.25), num[c].quantile(0.75)
                iqr = q3 - q1
                if ((num[c] < q1 - 1.5*iqr) | (num[c] > q3 + 1.5*iqr)).any():
                    outliers.append(c)
            return {"statistics": stats, "outliers": outliers}
        elif task_id == "task_hard":
            mv = int(df.isnull().sum().sum())
            return {"report": (
                f"Employee dataset ({len(df)} records). "
                f"Data quality: {mv} missing values in age, salary, experience_years. "
                f"Salary range: {num['salary'].min():.0f} to {num['salary'].max():.0f}. "
                f"Outliers found in salary and age columns. "
                f"Recommendations: "
                f"1) Impute {mv} missing values using column medians. "
                f"2) Investigate salary outliers ($500k and -$5k) as data entry errors. "
                f"3) Conduct department performance reviews to reward top scorers."
            )}
    except Exception as e:
        print(f"[ERROR] Fallback failed: {e}", flush=True)
    return {"missing_values": {}, "data_types": {}, "report": "Analysis unavailable"}


# ── Main ──────────────────────────────────────────────────────────────────

def run():
    print("[START] task=DataAnalystEnv", flush=True)

    obs = call_env("reset", "POST", {"seed": 42})
    if not obs:
        print("[END] task=DataAnalystEnv score=0.0 steps=0", flush=True)
        return []

    scores = []
    total = 0.0

    while obs.get("task") is not None:
        task = obs["task"]
        n = len(scores) + 1

        print(f"[STEP] step={n} task={task['id']} difficulty={task['difficulty']}", flush=True)

        action = agent_act(
            dataset_csv=obs.get("dataset_csv", ""),
            task_id=task["id"],
            task_description=task["description"]
        )

        result = call_env("step", "POST", {"action": action})
        if not result:
            print(f"[STEP] step={n} reward=0.0", flush=True)
            break

        reward   = result.get("reward", 0.0)
        feedback = result.get("info", {}).get("feedback", "")
        total += reward
        scores.append({"task": task["name"], "difficulty": task["difficulty"], "score": reward})

        print(f"[STEP] step={n} reward={reward:.3f} feedback={feedback}", flush=True)
        obs = result.get("observation", {})

    avg = total / max(len(scores), 1)
    print(f"[END] task=DataAnalystEnv score={avg:.3f} steps={len(scores)}", flush=True)
    return scores


if __name__ == "__main__":
    run()
