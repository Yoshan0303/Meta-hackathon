"""
Core EmailTriageEnv environment.

Follows the standard reset → step loop. Each episode presents the agent
with an inbox of emails (one at a time) and collects triage decisions.
The episode ends when all emails have been processed.
"""

from __future__ import annotations

from typing import Any

from email_triage_env.data.emails import GroundTruthEmail
from email_triage_env.models import EmailAction, EmailObservation, EmailReward
from email_triage_env.reward import compute_reward
from email_triage_env.tasks import TaskConfig, TrajectoryStep, get_task


class EmailTriageEnv:
    """
    Email triage and prioritization environment for AI agent training.

    Usage::

        env = EmailTriageEnv(task_id="basic_triage")
        obs = env.reset()

        while True:
            action = agent.act(obs)          # your agent here
            obs, reward, done, info = env.step(action)
            if done:
                break

        score = env.grade()
        print(f"Final score: {score:.2f}")
    """

    def __init__(self, task_id: str = "basic_triage", seed: int = 42) -> None:
        """
        Initialize the environment for a specific task.

        Parameters
        ----------
        task_id : str
            One of 'basic_triage', 'triage_with_replies',
            'full_triage_under_pressure'.
        seed : int
            Random seed for deterministic email generation.
        """
        self._task: TaskConfig = get_task(task_id, seed=seed)
        self._seed = seed

        # Episode state (set on reset)
        self._email_queue: list[GroundTruthEmail] = []
        self._current_index: int = 0
        self._trajectory: list[TrajectoryStep] = []
        self._cumulative_reward: float = 0.0
        self._running_priority_correct: int = 0
        self._done: bool = False
        self._reset_called: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> EmailObservation:
        """
        Reset the environment and return the first observation.

        A fresh copy of the task's email queue is loaded.
        """
        self._email_queue = list(self._task.emails)  # defensive copy
        self._current_index = 0
        self._trajectory = []
        self._cumulative_reward = 0.0
        self._running_priority_correct = 0
        self._done = False
        self._reset_called = True
        return self._make_observation()

    def step(self, action: EmailAction) -> tuple[EmailObservation, float, bool, dict[str, Any]]:
        """
        Submit a triage action for the current email and advance.

        Parameters
        ----------
        action : EmailAction
            The agent's triage decision for the current email.

        Returns
        -------
        observation : EmailObservation
            The next email (or the last email repeated if done).
        reward : float
            Step reward in [-1.0, 1.0].
        done : bool
            True when all emails have been processed.
        info : dict
            Diagnostic information including the EmailReward breakdown.
        """
        if not self._reset_called:
            raise RuntimeError("Must call reset() before step().")
        if self._done:
            raise RuntimeError("Episode is done. Call reset() to start a new one.")

        current_email = self._email_queue[self._current_index]

        # Compute reward
        step_reward, pri_ok, cat_ok, act_ok, rq = compute_reward(
            action_priority=action.priority,
            action_category=action.category,
            action_action=action.action,
            action_reply_text=action.reply_text,
            ground_truth=current_email,
            step_num=self._current_index,
            max_steps=len(self._email_queue),
            running_correct=self._running_priority_correct,
            running_total=self._current_index,
        )

        # Update running stats
        self._cumulative_reward += step_reward
        if pri_ok:
            self._running_priority_correct += 1

        # Record trajectory
        self._trajectory.append(TrajectoryStep(email=current_email, action=action))

        # Build reward model
        reward_info = EmailReward(
            step_reward=step_reward,
            cumulative_reward=self._cumulative_reward,
            priority_correct=pri_ok,
            category_correct=cat_ok,
            action_appropriate=act_ok,
            reply_quality=rq,
        )

        # Advance
        self._current_index += 1
        self._done = self._current_index >= len(self._email_queue)

        # Build next observation (repeat last if done)
        if self._done:
            next_obs = self._make_observation(index=self._current_index - 1, force_remaining=0)
        else:
            next_obs = self._make_observation()

        info: dict[str, Any] = {
            "reward_breakdown": reward_info.model_dump(),
            "step": self._current_index - 1,
            "done": self._done,
        }

        return next_obs, step_reward, self._done, info

    def state(self) -> dict[str, Any]:
        """
        Return the full current state of the environment.

        Useful for debugging and logging.
        """
        return {
            "task_id": self._task.task_id,
            "difficulty": self._task.difficulty,
            "max_steps": self._task.max_steps,
            "current_step": self._current_index,
            "emails_total": len(self._email_queue),
            "emails_remaining": max(0, len(self._email_queue) - self._current_index),
            "cumulative_reward": self._cumulative_reward,
            "trajectory_length": len(self._trajectory),
            "done": self._done,
            "priority_accuracy": (
                self._running_priority_correct / self._current_index
                if self._current_index > 0
                else 0.0
            ),
        }

    def grade(self) -> float:
        """
        Grade the completed trajectory using the task's deterministic grader.

        Returns a score in [0.0, 1.0]. Should only be called after the
        episode is done.
        """
        if not self._done:
            raise RuntimeError(
                "Cannot grade an incomplete episode. Run until done=True."
            )
        return self._task.grade(self._trajectory)

    @property
    def task(self) -> TaskConfig:
        """Access the current task configuration."""
        return self._task

    @property
    def trajectory(self) -> list[TrajectoryStep]:
        """Access the recorded trajectory."""
        return list(self._trajectory)

    @property
    def done(self) -> bool:
        """Whether the episode has ended."""
        return self._done

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_observation(
        self,
        index: int | None = None,
        force_remaining: int | None = None,
    ) -> EmailObservation:
        """Convert the current email into an EmailObservation."""
        idx = index if index is not None else self._current_index
        email = self._email_queue[idx]
        remaining = (
            force_remaining
            if force_remaining is not None
            else max(0, len(self._email_queue) - idx - 1)
        )
        return EmailObservation(
            email_id=email.email_id,
            subject=email.subject,
            sender=email.sender,
            sender_domain=email.sender_domain,
            body=email.body,
            timestamp=email.timestamp,
            thread_length=email.thread_length,
            has_attachment=email.has_attachment,
            step_number=idx,
            emails_remaining=remaining,
        )
