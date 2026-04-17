"""
loadEnv.py — loads and validates all required env vars from Practice/.env.

Raises EnvironmentError immediately at startup if any variable is missing or
has the wrong type — fail fast so the problem is obvious before the agent runs.
"""
import os
from dotenv import load_dotenv

load_dotenv()

def _require(key: str) -> str:
    value = os.getenv(key)
    if not value:
        raise EnvironmentError(f"Missing required env var: {key}  →  check Practice/.env")
    return value

def _require_int(key: str) -> int:
    value = _require(key)
    try:
        return int(value)
    except ValueError:
        raise EnvironmentError(
            f"Env var {key}={value!r} must be an integer  →  check Practice/.env"
        )

BASIC_MODEL          = _require("BASIC_MODEL")
ADVANCED_MODEL       = _require("ADVANCED_MODEL")
SYSTEM_PROMPT        = _require("SYSTEM_PROMPT")
CODE_PROMPT          = _require("CODE_PROMPT")
COMPLEXITY_THRESHOLD = _require_int("COMPLEXITY_THRESHOLD")
MAX_RETRIES          = _require_int("MAX_RETRIES")
