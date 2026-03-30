---
title: Data Analyst Env
emoji: 📊
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
tags:
  - openenv
---

# DataAnalystEnv 📊

An **OpenEnv** environment where an AI agent learns to perform real-world data analysis tasks on employee datasets.

## Tasks
| # | Task | Difficulty |
|---|------|-----------|
| 1 | Data Quality Inspector | Easy |
| 2 | Statistical Analyst | Medium |
| 3 | Insight Reporter | Hard |

## Setup
```bash
pip install -r requirements.txt
uvicorn server:app --host 0.0.0.0 --port 7860
```

## Environment Variables
| Variable | Description |
|----------|-------------|
| `API_BASE_URL` | LLM API endpoint |
| `MODEL_NAME` | Model identifier |
| `HF_TOKEN` | Hugging Face / API key |

## API Endpoints
- `POST /reset` — Start new episode
- `POST /step` — Submit action, get reward
- `GET /state` — Current environment state
- `GET /health` — Health check
