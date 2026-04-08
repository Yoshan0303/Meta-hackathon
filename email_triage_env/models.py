"""
EmailTriageEnv — Pydantic v2 models for Observation, Action, and Reward.

These models define the structured interfaces between the environment
and the agent, ensuring type safety and validation at every boundary.
"""

from typing import Literal, Optional

from pydantic import BaseModel, Field, model_validator


class EmailObservation(BaseModel):
    """
    What the agent sees at each step: a single email from the inbox.

    The agent receives no ground-truth labels — it must infer priority,
    category, and appropriate action from the email content alone.
    """

    email_id: str = Field(..., description="Unique identifier for this email")
    subject: str = Field(..., description="Email subject line")
    sender: str = Field(..., description="Sender display name")
    sender_domain: str = Field(..., description="Sender email domain (e.g. acme.com)")
    body: str = Field(..., description="Full email body text")
    timestamp: str = Field(..., description="ISO 8601 timestamp of email receipt")
    thread_length: int = Field(..., ge=1, description="Number of messages in this thread")
    has_attachment: bool = Field(..., description="Whether the email has attachments")
    step_number: int = Field(..., ge=0, description="Current step index (0-based)")
    emails_remaining: int = Field(..., ge=0, description="Emails left after this one")


class EmailAction(BaseModel):
    """
    The agent's triage decision for one email.

    If `action` is 'reply', the agent must provide `reply_text`.
    All other actions do not require reply text.
    """

    priority: Literal["urgent", "high", "medium", "low"] = Field(
        ..., description="Assessed priority level"
    )
    category: Literal["bug_report", "feature_request", "billing", "support", "spam"] = Field(
        ..., description="Email category classification"
    )
    action: Literal["reply", "escalate", "archive", "delete", "snooze"] = Field(
        ..., description="Triage action to take"
    )
    reply_text: Optional[str] = Field(
        default=None, description="Draft reply text (required when action='reply')"
    )

    @model_validator(mode="after")
    def validate_reply_text(self) -> "EmailAction":
        """Ensure reply_text is provided when action is 'reply'."""
        if self.action == "reply" and not self.reply_text:
            raise ValueError("reply_text is required when action is 'reply'")
        return self


class EmailReward(BaseModel):
    """
    Dense reward signal returned after each step.

    Provides both the scalar reward and detailed diagnostics so the
    agent (or researcher) can understand *why* a score was given.
    """

    step_reward: float = Field(
        ..., ge=-1.0, le=1.0, description="Reward for this step, clipped to [-1, 1]"
    )
    cumulative_reward: float = Field(
        ..., description="Running total reward across the episode"
    )
    priority_correct: bool = Field(
        ..., description="Whether the priority classification was exact-match correct"
    )
    category_correct: bool = Field(
        ..., description="Whether the category classification was exact-match correct"
    )
    action_appropriate: bool = Field(
        ..., description="Whether the chosen action was appropriate for this email"
    )
    reply_quality: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Quality score of the reply (0.0 if no reply needed)",
    )
