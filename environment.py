import json
import random
import pandas as pd
import numpy as np
from typing import Any
from models import Observation, StepResult, TaskInfo

TASKS = [
    {
        "id": "task_easy",
        "name": "Data Quality Inspector",
        "difficulty": "easy",
        "description": "Analyze the dataset and identify: (1) columns with missing values and their counts, (2) data type of each column.",
        "expected_keys": ["missing_values", "data_types"]
    },
    {
        "id": "task_medium",
        "name": "Statistical Analyst",
        "difficulty": "medium",
        "description": "Provide: (1) summary statistics (mean, median, std) for all numeric columns, (2) list of outlier column names using IQR method.",
        "expected_keys": ["statistics", "outliers"]
    },
    {
        "id": "task_hard",
        "name": "Insight Reporter",
        "difficulty": "hard",
        "description": "Write a comprehensive data analysis report including: key findings, data quality issues, trends, and 3 actionable business recommendations.",
        "expected_keys": ["report"]
    }
]

def generate_dataset(seed: int = 42) -> pd.DataFrame:
    np.random.seed(seed)
    n = 100
    df = pd.DataFrame({
        "age": np.random.randint(18, 65, n).astype(float),
        "salary": np.random.normal(50000, 15000, n),
        "experience_years": np.random.randint(0, 30, n).astype(float),
        "department": np.random.choice(["Sales", "Engineering", "HR", "Marketing"], n),
        "performance_score": np.random.uniform(1, 10, n),
        "tenure_months": np.random.randint(1, 120, n).astype(float),
    })
    # Inject missing values
    df.loc[np.random.choice(n, 10, replace=False), "age"] = np.nan
    df.loc[np.random.choice(n, 7, replace=False), "salary"] = np.nan
    df.loc[np.random.choice(n, 5, replace=False), "experience_years"] = np.nan
    # Inject outliers
    df.loc[0, "salary"] = 500000
    df.loc[1, "salary"] = -5000
    df.loc[2, "age"] = 150
    return df

class DataAnalystEnv:
    def __init__(self):
        self._df: pd.DataFrame = None
        self._current_task_idx: int = 0
        self._done: bool = False
        self._step_count: int = 0
        self._seed: int = 42

    def reset(self, seed: int = None) -> Observation:
        self._seed = seed if seed is not None else random.randint(0, 9999)
        self._df = generate_dataset(self._seed)
        self._current_task_idx = 0
        self._done = False
        self._step_count = 0
        return self._make_observation()

    def step(self, action: dict) -> StepResult:
        if self._done:
            raise RuntimeError("Episode is done. Call reset() first.")

        task = TASKS[self._current_task_idx]
        reward, feedback = self._grade(task, action)

        self._step_count += 1
        self._current_task_idx += 1
        if self._current_task_idx >= len(TASKS):
            self._done = True

        obs = self._make_observation()
        return StepResult(
            observation=obs,
            reward=reward,
            done=self._done,
            info={"task_id": task["id"], "feedback": feedback, "step": self._step_count}
        )

    def state(self) -> dict:
        return {
            "current_task_idx": self._current_task_idx,
            "step_count": self._step_count,
            "done": self._done,
            "dataset_shape": list(self._df.shape) if self._df is not None else None,
            "seed": self._seed
        }

    def _make_observation(self) -> Observation:
        if self._done:
            return Observation(
                dataset_csv=None,
                task=None,
                message="All tasks completed! Episode done."
            )
        task = TASKS[self._current_task_idx]
        return Observation(
            dataset_csv=self._df.to_csv(index=False),
            task=TaskInfo(
                id=task["id"],
                name=task["name"],
                difficulty=task["difficulty"],
                description=task["description"]
            ),
            message=f"Task {self._current_task_idx + 1} of {len(TASKS)}: {task['name']}"
        )

    def _grade(self, task: dict, action: dict) -> tuple[float, str]:
        if task["id"] == "task_easy":
            return self._grade_easy(action)
        elif task["id"] == "task_medium":
            return self._grade_medium(action)
        elif task["id"] == "task_hard":
            return self._grade_hard(action)
        return 0.0, "Unknown task"

    def _grade_easy(self, action: dict) -> tuple[float, str]:
        score = 0.0
        feedback = []

        # Check missing_values key
        mv = action.get("missing_values", {})
        if isinstance(mv, dict):
            true_missing = self._df.isnull().sum()
            true_missing = {k: int(v) for k, v in true_missing.items() if v > 0}
            correct_cols = sum(1 for col in true_missing if col in mv)
            score += 0.4 * (correct_cols / max(len(true_missing), 1))
            if correct_cols == len(true_missing):
                feedback.append("✓ All missing value columns identified")
            else:
                feedback.append(f"✗ Found {correct_cols}/{len(true_missing)} missing value columns")
        else:
            feedback.append("✗ missing_values should be a dict")

        # Check data_types key
        dt = action.get("data_types", {})
        if isinstance(dt, dict):
            correct_types = sum(1 for col in self._df.columns if col in dt)
            score += 0.6 * (correct_types / len(self._df.columns))
            feedback.append(f"✓ {correct_types}/{len(self._df.columns)} column types identified" if correct_types == len(self._df.columns) else f"✗ Only {correct_types}/{len(self._df.columns)} types found")
        else:
            feedback.append("✗ data_types should be a dict")

        return round(min(score, 1.0), 3), " | ".join(feedback)

    def _grade_medium(self, action: dict) -> tuple[float, str]:
        score = 0.0
        feedback = []
        numeric_cols = self._df.select_dtypes(include=np.number).columns.tolist()

        # Check statistics
        stats = action.get("statistics", {})
        if isinstance(stats, dict):
            covered = sum(1 for col in numeric_cols if col in stats)
            col_score = covered / max(len(numeric_cols), 1)
            key_score = 0.0
            for col in numeric_cols:
                if col in stats:
                    col_stats = stats[col]
                    if isinstance(col_stats, dict):
                        has_keys = sum(1 for k in ["mean", "median", "std"] if k in col_stats)
                        key_score += has_keys / 3
            key_score /= max(len(numeric_cols), 1)
            score += 0.5 * (col_score * 0.4 + key_score * 0.6)
            feedback.append(f"✓ Stats for {covered}/{len(numeric_cols)} numeric columns")
        else:
            feedback.append("✗ statistics should be a dict")

        # Check outliers using IQR
        outliers = action.get("outliers", [])
        if isinstance(outliers, list):
            true_outlier_cols = []
            for col in numeric_cols:
                q1 = self._df[col].quantile(0.25)
                q3 = self._df[col].quantile(0.75)
                iqr = q3 - q1
                has_outlier = ((self._df[col] < q1 - 1.5 * iqr) | (self._df[col] > q3 + 1.5 * iqr)).any()
                if has_outlier:
                    true_outlier_cols.append(col)
            correct = len(set(outliers) & set(true_outlier_cols))
            precision = correct / max(len(outliers), 1)
            recall = correct / max(len(true_outlier_cols), 1)
            f1 = 2 * precision * recall / max(precision + recall, 1e-9)
            score += 0.5 * f1
            feedback.append(f"✓ Outlier detection F1={f1:.2f}" if f1 > 0.5 else f"✗ Outlier detection F1={f1:.2f}")
        else:
            feedback.append("✗ outliers should be a list")

        return round(min(score, 1.0), 3), " | ".join(feedback)

    def _grade_hard(self, action: dict) -> tuple[float, str]:
        """LLM-as-judge grader for open-ended report."""
        import os
        from openai import OpenAI

        report = action.get("report", "")
        if not report or len(report.strip()) < 100:
            return 0.1, "✗ Report too short or missing"

        client = OpenAI(
            base_url=os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1"),
            api_key=os.environ.get("HF_TOKEN") or os.environ.get("OPENAI_API_KEY", "")
        )
        prompt = f"""You are a data analysis expert. Grade the following report on a scale from 0.0 to 1.0.

Criteria:
- Data quality issues mentioned (0.0-0.25): Does it mention missing values, outliers, or data problems?
- Statistical insights (0.0-0.25): Are numeric trends, distributions, or patterns discussed?
- Business relevance (0.0-0.25): Are the findings tied to business context?
- Recommendations (0.0-0.25): Are there 3 specific, actionable recommendations?

Dataset context: Employee dataset with age, salary, experience_years, department, performance_score, tenure_months.

Report to grade:
\"\"\"
{report[:2000]}
\"\"\"

Respond with ONLY a JSON object: {{"score": <float 0.0-1.0>, "reason": "<one sentence>"}}"""

        try:
            response = client.chat.completions.create(
                model=os.environ.get("MODEL_NAME", "gpt-4o-mini"),
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
                temperature=0
            )
            raw = response.choices[0].message.content.strip()
            parsed = json.loads(raw)
            score = float(parsed.get("score", 0.5))
            reason = parsed.get("reason", "")
            return round(min(max(score, 0.0), 1.0), 3), f"LLM Judge: {reason}"
        except Exception as e:
            # Fallback: heuristic scoring
            score = 0.0
            checks = ["missing", "outlier", "recommend", "salary", "performance", "trend"]
            hits = sum(1 for c in checks if c.lower() in report.lower())
            score = hits / len(checks)
            return round(score, 3), f"Heuristic grade (LLM unavailable): {hits}/{len(checks)} key topics covered"
