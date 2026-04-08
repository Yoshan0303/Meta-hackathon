"""EmailTriageEnv — A real-world email triage environment for AI agent training."""

from email_triage_env.environment import EmailTriageEnv
from email_triage_env.models import EmailAction, EmailObservation, EmailReward

__all__ = ["EmailTriageEnv", "EmailObservation", "EmailAction", "EmailReward"]
__version__ = "1.0.0"
