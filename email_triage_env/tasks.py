"""
Task definitions and deterministic graders for EmailTriageEnv.

Three tasks at increasing difficulty:
  1. basic_triage          (easy)   — 20 clear-signal emails
  2. triage_with_replies   (medium) — 20 emails, 8 require replies
  3. full_triage_under_pressure (hard) — 30 emails with adversarial cases

Each task returns a float score in [0.0, 1.0] from its grade() method.
Graders are fully deterministic — no LLM calls.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from email_triage_env.data.emails import (
    GroundTruthEmail,
    get_easy_emails,
    get_hard_emails,
    get_medium_emails,
)
from email_triage_env.models import EmailAction
from email_triage_env.reward import _reply_quality


# ---------------------------------------------------------------------------
# Trajectory entry — one step recorded during an episode
# ---------------------------------------------------------------------------

@dataclass
class TrajectoryStep:
    """A single (email, action) pair from an episode trajectory."""

    email: GroundTruthEmail
    action: EmailAction


# ---------------------------------------------------------------------------
# Base task config
# ---------------------------------------------------------------------------

@dataclass
class TaskConfig:
    """Configuration for a single evaluation task."""

    task_id: str
    description: str
    difficulty: Literal["easy", "medium", "hard"]
    max_steps: int
    emails: list[GroundTruthEmail] = field(default_factory=list)

    def grade(self, trajectory: list[TrajectoryStep]) -> float:
        """
        Grade a completed trajectory.

        Must be overridden by subclasses. Returns a float in [0.0, 1.0].
        """
        raise NotImplementedError


# ---------------------------------------------------------------------------
# TASK 1 — basic_triage (easy)
# ---------------------------------------------------------------------------

class BasicTriageTask(TaskConfig):
    """
    Easy task: 20 non-ambiguous emails.

    Score = (priority_correct + category_correct) / (2 * total_emails)

    Expected baseline: ~0.75
    """

    def __init__(self, seed: int = 42) -> None:
        emails = get_easy_emails(seed)
        super().__init__(
            task_id="basic_triage",
            description=(
                "Classify 20 clearly-worded emails by priority and category. "
                "No ambiguity — signals are obvious in the subject and body."
            ),
            difficulty="easy",
            max_steps=20,
            emails=emails,
        )

    def grade(self, trajectory: list[TrajectoryStep]) -> float:
        """
        Score = (priority_correct + category_correct) / (2 * total_emails)
        """
        if not trajectory:
            return 0.0

        total = len(trajectory)
        priority_hits = sum(
            1 for step in trajectory
            if step.action.priority == step.email.true_priority
        )
        category_hits = sum(
            1 for step in trajectory
            if step.action.category == step.email.true_category
        )
        return (priority_hits + category_hits) / (2 * total)


# ---------------------------------------------------------------------------
# TASK 2 — triage_with_replies (medium)
# ---------------------------------------------------------------------------

class TriageWithRepliesTask(TaskConfig):
    """
    Medium task: 20 emails, 8 of which require replies.

    Score = 0.4 * priority_acc + 0.3 * category_acc + 0.3 * avg_reply_quality

    Expected baseline: ~0.55
    """

    def __init__(self, seed: int = 42) -> None:
        emails = get_medium_emails(seed)
        super().__init__(
            task_id="triage_with_replies",
            description=(
                "Classify 20 emails by priority and category. For 8 emails that "
                "require a reply, draft a relevant response. Reply quality is "
                "scored by keyword relevance."
            ),
            difficulty="medium",
            max_steps=20,
            emails=emails,
        )

    def grade(self, trajectory: list[TrajectoryStep]) -> float:
        """
        Score = 0.4 * priority_acc + 0.3 * category_acc + 0.3 * avg_reply_quality
        """
        if not trajectory:
            return 0.0

        total = len(trajectory)
        priority_acc = sum(
            1 for s in trajectory
            if s.action.priority == s.email.true_priority
        ) / total

        category_acc = sum(
            1 for s in trajectory
            if s.action.category == s.email.true_category
        ) / total

        # Reply quality for emails that expect a reply
        reply_emails = [s for s in trajectory if s.email.expected_action == "reply"]
        if reply_emails:
            reply_scores = []
            for s in reply_emails:
                if s.action.action == "reply" and s.action.reply_text:
                    rq = _reply_quality(s.action.reply_text, s.email.reply_keywords)
                else:
                    rq = 0.0  # didn't reply when expected → 0 quality
                reply_scores.append(rq)
            avg_reply_quality = sum(reply_scores) / len(reply_scores)
        else:
            avg_reply_quality = 0.0

        score = 0.4 * priority_acc + 0.3 * category_acc + 0.3 * avg_reply_quality
        return min(max(score, 0.0), 1.0)


# ---------------------------------------------------------------------------
# TASK 3 — full_triage_under_pressure (hard)
# ---------------------------------------------------------------------------

class FullTriageUnderPressureTask(TaskConfig):
    """
    Hard task: 30 emails with adversarial cases.

    Includes 5 escalation-worthy emails. Penalizes false escalations
    heavily (-0.1 each, normalized).

    Score = complex weighted formula:
        base  = 0.30 * priority_acc + 0.20 * category_acc
                + 0.20 * action_acc + 0.15 * escalation_precision
                + 0.15 * avg_reply_quality
        penalty = 0.10 * false_escalation_count
        score = max(0, base - penalty)

    Expected baseline: ~0.35
    """

    def __init__(self, seed: int = 42) -> None:
        emails = get_hard_emails(seed)
        super().__init__(
            task_id="full_triage_under_pressure",
            description=(
                "Handle 30 emails including adversarial cases: misleading subjects, "
                "disguised spam, and urgent-looking but low-priority emails. "
                "Correctly identify 5 escalation-worthy emails while avoiding "
                "false escalations."
            ),
            difficulty="hard",
            max_steps=30,
            emails=emails,
        )

    def grade(self, trajectory: list[TrajectoryStep]) -> float:
        """Complex weighted grader with escalation precision and heavy penalties."""
        if not trajectory:
            return 0.0

        total = len(trajectory)

        # --- Priority accuracy ---
        priority_acc = sum(
            1 for s in trajectory
            if s.action.priority == s.email.true_priority
        ) / total

        # --- Category accuracy ---
        category_acc = sum(
            1 for s in trajectory
            if s.action.category == s.email.true_category
        ) / total

        # --- Action accuracy ---
        action_acc = sum(
            1 for s in trajectory
            if s.action.action == s.email.expected_action
        ) / total

        # --- Escalation precision ---
        # Emails that truly warrant escalation
        true_escalations = {
            s.email.email_id
            for s in trajectory
            if s.email.expected_action == "escalate"
        }
        # Emails the agent chose to escalate
        agent_escalations = {
            s.email.email_id
            for s in trajectory
            if s.action.action == "escalate"
        }

        # Precision: of the emails the agent escalated, how many were correct?
        if agent_escalations:
            true_positives = len(agent_escalations & true_escalations)
            escalation_precision = true_positives / len(agent_escalations)
        else:
            # Agent didn't escalate anything — 0 precision
            escalation_precision = 0.0

        # Count false escalations (escalated but shouldn't have been)
        false_escalations = len(agent_escalations - true_escalations)

        # --- Reply quality ---
        reply_emails = [s for s in trajectory if s.email.expected_action == "reply"]
        if reply_emails:
            reply_scores = []
            for s in reply_emails:
                if s.action.action == "reply" and s.action.reply_text:
                    rq = _reply_quality(s.action.reply_text, s.email.reply_keywords)
                else:
                    rq = 0.0
                reply_scores.append(rq)
            avg_reply_quality = sum(reply_scores) / len(reply_scores)
        else:
            avg_reply_quality = 0.0

        # --- Compute final score ---
        base = (
            0.30 * priority_acc
            + 0.20 * category_acc
            + 0.20 * action_acc
            + 0.15 * escalation_precision
            + 0.15 * avg_reply_quality
        )
        penalty = 0.10 * false_escalations
        score = max(0.0, base - penalty)
        return min(score, 1.0)


# ---------------------------------------------------------------------------
# Task registry
# ---------------------------------------------------------------------------

def get_task(task_id: str, seed: int = 42) -> TaskConfig:
    """Look up a task by its ID and return a fresh instance."""
    registry: dict[str, type[TaskConfig]] = {
        "basic_triage": BasicTriageTask,
        "triage_with_replies": TriageWithRepliesTask,
        "full_triage_under_pressure": FullTriageUnderPressureTask,
    }
    cls = registry.get(task_id)
    if cls is None:
        available = ", ".join(sorted(registry.keys()))
        raise ValueError(f"Unknown task_id '{task_id}'. Available: {available}")
    return cls(seed=seed)


AVAILABLE_TASKS: list[str] = [
    "basic_triage",
    "triage_with_replies",
    "full_triage_under_pressure",
]
