"""
OpenEnv HTTP server for EmailTriageEnv.

Exposes the environment via REST API endpoints:
  POST /reset   — Reset the environment, returns initial observation
  POST /step    — Submit an action, returns (observation, reward, done, info)
  POST /grade   — Grade the completed episode
  GET  /state   — Get current environment state
  GET  /health  — Health check

Reads configuration from environment variables:
  TASK_ID  — Task to run (default: basic_triage)
  SEED     — Random seed (default: 42)
  PORT     — Server port (default: 8000)
"""

from __future__ import annotations

import json
import os
import sys
from http.server import HTTPServer, BaseHTTPRequestHandler

# Add project to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from email_triage_env.environment import EmailTriageEnv
from email_triage_env.models import EmailAction


class EnvironmentState:
    """Holds the global environment instance."""
    env: EmailTriageEnv | None = None
    task_id: str = "basic_triage"
    seed: int = 42


state = EnvironmentState()


class OpenEnvHandler(BaseHTTPRequestHandler):
    """HTTP request handler for OpenEnv API."""

    def _send_json(self, data: dict, status: int = 200) -> None:
        """Send a JSON response."""
        body = json.dumps(data, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict:
        """Read JSON from request body."""
        content_length = int(self.headers.get("Content-Length", 0))
        if content_length == 0:
            return {}
        raw = self.rfile.read(content_length)
        return json.loads(raw.decode("utf-8"))

    def do_GET(self) -> None:
        """Handle GET requests."""
        if self.path == "/health":
            self._send_json({"status": "ok"})
        elif self.path == "/state":
            if state.env is None:
                self._send_json({"error": "Environment not initialized. Call POST /reset first."}, 400)
                return
            self._send_json(state.env.state())
        else:
            self._send_json({"error": f"Unknown endpoint: {self.path}"}, 404)

    def do_POST(self) -> None:
        """Handle POST requests."""
        if self.path == "/reset":
            self._handle_reset()
        elif self.path == "/step":
            self._handle_step()
        elif self.path == "/grade":
            self._handle_grade()
        else:
            self._send_json({"error": f"Unknown endpoint: {self.path}"}, 404)

    def _handle_reset(self) -> None:
        """POST /reset — Reset environment and return initial observation."""
        try:
            body = self._read_json()
            task_id = body.get("task_id", state.task_id)
            seed = body.get("seed", state.seed)

            state.env = EmailTriageEnv(task_id=task_id, seed=seed)
            obs = state.env.reset()

            self._send_json({
                "observation": obs.model_dump(),
            })
        except Exception as e:
            self._send_json({"error": str(e)}, 500)

    def _handle_step(self) -> None:
        """POST /step — Submit action, return next observation + reward."""
        if state.env is None:
            self._send_json({"error": "Environment not initialized. Call POST /reset first."}, 400)
            return

        try:
            body = self._read_json()

            action = EmailAction(
                priority=body.get("priority", "low"),
                category=body.get("category", "support"),
                action=body.get("action", "archive"),
                reply_text=body.get("reply_text"),
            )

            obs, reward, done, info = state.env.step(action)

            self._send_json({
                "observation": obs.model_dump(),
                "reward": reward,
                "done": done,
                "info": info,
            })
        except Exception as e:
            self._send_json({"error": str(e)}, 500)

    def _handle_grade(self) -> None:
        """POST /grade — Grade the completed episode."""
        if state.env is None:
            self._send_json({"error": "Environment not initialized. Call POST /reset first."}, 400)
            return

        try:
            score = state.env.grade()
            self._send_json({"score": score})
        except Exception as e:
            self._send_json({"error": str(e)}, 500)

    def log_message(self, format, *args):
        """Override to use standard print for logging."""
        print(f"[OpenEnv] {args[0]} {args[1]} {args[2]}")


def main() -> None:
    """Start the OpenEnv HTTP server."""
    port = int(os.environ.get("PORT", 8000))
    server = HTTPServer(("0.0.0.0", port), OpenEnvHandler)
    print(f"OpenEnv server running on http://0.0.0.0:{port}")
    print(f"Endpoints: POST /reset, POST /step, POST /grade, GET /state, GET /health")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server...")
        server.server_close()


if __name__ == "__main__":
    main()
