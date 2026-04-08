# 📧 EmailTriageEnv

A production-ready OpenEnv environment for training and evaluating AI agents on real-world email triage and prioritization tasks.

---

## 1. Overview & Motivation

Email triage is one of the most common knowledge-work tasks: prioritize incoming messages, categorize them, decide on the right action, and draft replies when needed. It's also deceptively hard for AI — it requires:

- **Classification under ambiguity** — subject lines can be misleading, tone varies wildly
- **Multi-label decision-making** — priority, category, and action are interdependent
- **Natural language generation** — replies must be relevant, professional, and concise
- **Adversarial robustness** — spam disguised as legitimate emails, casual urgency, false alarms

EmailTriageEnv provides a structured, reproducible environment where AI agents can learn and be benchmarked on all of these skills simultaneously.

**Context:** The environment simulates the support inbox of "CloudMetrics," a mid-size B2B SaaS analytics company. Emails come from enterprise customers, internal teams, and spammers.

---

## 2. Environment Description

The agent receives emails one at a time from a simulated inbox and must:

1. **Classify priority** — `urgent` / `high` / `medium` / `low`
2. **Assign category** — `bug_report` / `feature_request` / `billing` / `support` / `spam`
3. **Decide action** — `reply` / `escalate` / `archive` / `delete` / `snooze`
4. **Draft a reply** (when action = `reply`)

An episode ends when all emails have been processed. The agent is scored by a deterministic grader — no LLM calls in the evaluation loop.

---

## 3. Observation Space

Each step, the agent receives an `EmailObservation` with these fields:

| Field              | Type   | Description                                      |
|--------------------|--------|--------------------------------------------------|
| `email_id`         | `str`  | Unique identifier for this email                 |
| `subject`          | `str`  | Email subject line                               |
| `sender`           | `str`  | Sender display name                              |
| `sender_domain`    | `str`  | Sender email domain (e.g., `acme.com`)            |
| `body`             | `str`  | Full email body text                             |
| `timestamp`        | `str`  | ISO 8601 timestamp                               |
| `thread_length`    | `int`  | Number of messages in the email thread           |
| `has_attachment`    | `bool` | Whether the email has file attachments            |
| `step_number`      | `int`  | Current step index (0-based)                     |
| `emails_remaining` | `int`  | Number of emails left after this one             |

---

## 4. Action Space

The agent must return an `EmailAction` for each email:

| Field        | Type                                                            | Description                                |
|--------------|------------------------------------------------------------------|--------------------------------------------|
| `priority`   | `Literal["urgent", "high", "medium", "low"]`                    | Assessed priority level                    |
| `category`   | `Literal["bug_report", "feature_request", "billing", "support", "spam"]` | Email category classification             |
| `action`     | `Literal["reply", "escalate", "archive", "delete", "snooze"]`   | Triage action to take                      |
| `reply_text` | `Optional[str]`                                                 | Draft reply (required when action="reply") |

---

## 5. Reward Function

EmailTriageEnv uses a **dense reward signal** — the agent receives informative feedback at every step, not just at the end. This accelerates learning.

| Component              | Reward      | Condition                                    |
|------------------------|-------------|----------------------------------------------|
| Priority exact match   | **+0.30**   | Agent's priority = ground truth              |
| Priority adjacent      | **+0.10**   | Agent is off by one level (e.g., high vs medium) |
| Category exact match   | **+0.30**   | Agent's category = ground truth              |
| Action appropriate     | **+0.20**   | Action is in the set of acceptable actions   |
| Reply quality          | **+0.20**   | Keyword overlap score (0–1) × 0.20           |
| False escalation       | **−0.30**   | Escalated a low-priority email               |
| Deleting urgent email  | **−0.20**   | Deleted an urgent email                      |
| Missing reply          | **−0.10**   | Didn't reply when a reply was expected       |
| Consistency bonus      | **+0.05**   | Maintaining ≥70% accuracy across the episode |

Step rewards are clipped to **[-1.0, 1.0]**.

---

## 6. Tasks

| Task ID                       | Difficulty | Emails | Description                                                    | Baseline Score |
|-------------------------------|------------|--------|----------------------------------------------------------------|----------------|
| `basic_triage`                | Easy       | 20     | Clear signals, no ambiguity. Priority + category accuracy.     | ~0.75          |
| `triage_with_replies`         | Medium     | 20     | 8 emails need replies. Weighted: priority, category, reply quality. | ~0.55          |
| `full_triage_under_pressure`  | Hard       | 30     | Adversarial cases, 5 escalations, heavy penalties for false escalation. | ~0.35          |

Each task has a **deterministic grader** — no LLM calls, fully reproducible scoring.

---

## 7. Installation & Setup

```bash
# Clone the repository
git clone <your-repo-url>
cd email_triage_env

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## 8. Quick Start

```python
from email_triage_env.environment import EmailTriageEnv
from email_triage_env.models import EmailAction

# Initialize environment
env = EmailTriageEnv(task_id="basic_triage", seed=42)

# Reset to get the first email
obs = env.reset()

while True:
    # Your agent logic here — this is a simple heuristic example
    action = EmailAction(
        priority="medium",
        category="support",
        action="archive",
        reply_text=None,
    )

    # Step the environment
    obs, reward, done, info = env.step(action)
    print(f"Step {info['step']}: reward={reward:+.2f}, done={done}")

    if done:
        break

# Grade the episode
score = env.grade()
print(f"Final score: {score:.2f}")

# Inspect environment state
print(env.state())
```

---

## 9. Running Baseline

The baseline script uses Google's `gemini-2.0-flash` model:

```bash
# Set your API key (from Google AI Studio)
set GOOGLE_API_KEY=AIza...

# Run the baseline
python -m email_triage_env.baseline
```

Expected output format:

```
Task 1 (easy)   -- basic_triage:           Score = 0.XX
Task 2 (medium) -- triage_with_replies:    Score = 0.XX
Task 3 (hard)   -- full_triage_under_pressure: Score = 0.XX
Average Score: 0.XX
```

---

## 10. Docker Usage

```bash
# Build the container
docker build -t email-triage-env .

# Run with your API key
docker run -e GOOGLE_API_KEY="AIza..." email-triage-env

# Interactive shell
docker run -it -e GOOGLE_API_KEY="AIza..." email-triage-env /bin/bash
```

---

## 11. HuggingFace Deployment

To deploy on HuggingFace Spaces:

1. Create a new Space (Docker SDK).
2. Upload all project files.
3. Set `GOOGLE_API_KEY` as a Space secret.
4. The Dockerfile will auto-build and run the baseline.

For a Gradio/Streamlit interface, wrap the environment in a UI:

```python
import gradio as gr
from email_triage_env.environment import EmailTriageEnv

env = EmailTriageEnv(task_id="basic_triage")
obs = env.reset()

def triage(priority, category, action, reply_text):
    from email_triage_env.models import EmailAction
    act = EmailAction(
        priority=priority,
        category=category,
        action=action,
        reply_text=reply_text if action == "reply" else None,
    )
    obs, reward, done, info = env.step(act)
    return obs.model_dump(), reward, done

# Build your Gradio interface around the triage function
```

---

## 12. OpenEnv Spec Compliance

This environment follows the OpenEnv specification:

| Requirement                    | Status |
|--------------------------------|--------|
| `openenv.yaml` metadata file  | ✅      |
| Pydantic observation model     | ✅      |
| Pydantic action model          | ✅      |
| Structured reward signal       | ✅      |
| Multiple difficulty tasks      | ✅      |
| Deterministic graders          | ✅      |
| Reproducible seeds             | ✅      |
| Baseline agent included        | ✅      |
| Docker support                 | ✅      |
| `reset()` / `step()` API       | ✅      |
| Reward clipped to [-1, 1]      | ✅      |

---

## Project Structure

```
email_triage_env/
├── __init__.py           # Package exports
├── environment.py        # Core EmailTriageEnv class
├── models.py             # Pydantic v2 models (Observation, Action, Reward)
├── tasks.py              # 3 tasks with deterministic graders
├── reward.py             # Dense reward shaping logic
├── data/
│   ├── __init__.py
│   └── emails.py         # Synthetic email dataset (200 emails, seed=42)
├── baseline.py           # Baseline inference (Google Gemini 2.0 Flash)
├── openenv.yaml          # OpenEnv metadata spec
├── Dockerfile            # Docker container
├── requirements.txt      # Python dependencies
└── README.md             # This file
```

---

## License

MIT
