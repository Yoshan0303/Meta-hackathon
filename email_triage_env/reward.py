"""
Dense reward function for EmailTriageEnv.

Computes a rich, informative reward signal at every step so the agent
gets immediate feedback on each triage decision.  The reward is clipped
to [-1.0, 1.0] and composed of additive components:

  +0.30  exact priority match  (or +0.10 for adjacent level)
  +0.30  exact category match
  +0.20  appropriate action choice
  +0.20  reply quality (keyword overlap; 0 when no reply expected)
  -0.30  false escalation of a low-priority email
  -0.20  deleting an urgent email
  -0.10  missing reply when one was expected
  +0.05  time-consistency bonus (stable accuracy across episode)
"""

from __future__ import annotations

import numpy as np

from email_triage_env.data.emails import GroundTruthEmail

# Maps priority label → numeric level for "adjacent" calculation
_PRIORITY_ORD: dict[str, int] = {
    "urgent": 3,
    "high": 2,
    "medium": 1,
    "low": 0,
}

# Action appropriateness lookup.
# Key  = (true_priority, true_category)
# Value = set of acceptable actions.
_APPROPRIATE_ACTIONS: dict[tuple[str, str], set[str]] = {
    # Urgent emails should be escalated or replied to
    ("urgent", "bug_report"): {"escalate", "reply"},
    ("urgent", "feature_request"): {"escalate", "reply"},
    ("urgent", "billing"): {"escalate", "reply"},
    ("urgent", "support"): {"escalate", "reply"},
    ("urgent", "spam"): {"delete"},  # edge case
    # High-priority emails need a reply or escalation
    ("high", "bug_report"): {"reply", "escalate"},
    ("high", "feature_request"): {"reply"},
    ("high", "billing"): {"reply", "escalate"},
    ("high", "support"): {"reply", "escalate"},
    ("high", "spam"): {"delete"},
    # Medium-priority emails — reply or archive
    ("medium", "bug_report"): {"reply", "archive"},
    ("medium", "feature_request"): {"reply", "archive"},
    ("medium", "billing"): {"reply"},
    ("medium", "support"): {"reply", "archive"},
    ("medium", "spam"): {"delete"},
    # Low-priority emails — archive, delete, or snooze
    ("low", "bug_report"): {"archive", "snooze"},
    ("low", "feature_request"): {"archive", "snooze"},
    ("low", "billing"): {"reply", "archive"},
    ("low", "support"): {"archive", "snooze", "reply"},
    ("low", "spam"): {"delete"},
}


def _reply_quality(reply_text: str | None, keywords: list[str]) -> float:
    """
    Score reply quality via keyword overlap.

    Returns a float in [0.0, 1.0].  If there are no expected keywords,
    or no reply was given, returns 0.0.
    """
    if not reply_text or not keywords:
        return 0.0

    reply_lower = reply_text.lower()
    hits = sum(1 for kw in keywords if kw.lower() in reply_lower)
    return min(hits / max(len(keywords), 1), 1.0)


def compute_reward(
    action_priority: str,
    action_category: str,
    action_action: str,
    action_reply_text: str | None,
    ground_truth: GroundTruthEmail,
    step_num: int,
    max_steps: int,
    running_correct: int = 0,
    running_total: int = 0,
) -> tuple[float, bool, bool, bool, float]:
    """
    Compute the dense reward for a single triage step.

    Parameters
    ----------
    action_priority : str
        Agent's predicted priority.
    action_category : str
        Agent's predicted category.
    action_action : str
        Agent's chosen action.
    action_reply_text : str | None
        Agent's reply draft (may be None).
    ground_truth : GroundTruthEmail
        The email's baked-in ground truth.
    step_num : int
        Current step index (0-based).
    max_steps : int
        Total number of steps in this episode.
    running_correct : int
        Number of previous steps where priority was correct.
    running_total : int
        Number of previous steps completed.

    Returns
    -------
    tuple of (reward, priority_correct, category_correct,
              action_appropriate, reply_quality_score)
    """
    reward = 0.0

    # --- Priority correctness: +0.3 exact, +0.1 adjacent ---
    priority_correct = action_priority == ground_truth.true_priority
    if priority_correct:
        reward += 0.30
    else:
        agent_ord = _PRIORITY_ORD.get(action_priority, -1)
        truth_ord = _PRIORITY_ORD.get(ground_truth.true_priority, -1)
        if abs(agent_ord - truth_ord) == 1:
            reward += 0.10  # adjacent level — partial credit

    # --- Category correctness: +0.3 ---
    category_correct = action_category == ground_truth.true_category
    if category_correct:
        reward += 0.30

    # --- Action appropriateness: +0.2 ---
    key = (ground_truth.true_priority, ground_truth.true_category)
    acceptable = _APPROPRIATE_ACTIONS.get(key, {ground_truth.expected_action})
    action_appropriate = action_action in acceptable
    if action_appropriate:
        reward += 0.20

    # --- Reply quality: +0.2 (only when reply expected) ---
    reply_expected = ground_truth.expected_action == "reply"
    rq = 0.0
    if reply_expected:
        rq = _reply_quality(action_reply_text, ground_truth.reply_keywords)
        reward += 0.20 * rq
    else:
        rq = 0.0

    # --- Penalties ---
    # Escalating a low-priority email is costly
    if action_action == "escalate" and ground_truth.true_priority == "low":
        reward -= 0.30

    # Deleting an urgent email is very bad
    if action_action == "delete" and ground_truth.true_priority == "urgent":
        reward -= 0.20

    # Missing reply when one was expected
    if reply_expected and action_action != "reply":
        reward -= 0.10

    # --- Time-consistency bonus ---
    # Small bonus for maintaining accuracy across the episode
    if running_total > 0:
        accuracy_so_far = running_correct / running_total
        if accuracy_so_far >= 0.7:
            reward += 0.05

    return (
        float(np.clip(reward, -1.0, 1.0)),
        priority_correct,
        category_correct,
        action_appropriate,
        rq,
    )
