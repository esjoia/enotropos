"""enotropos — Shared LLM client helpers.

Provides a single entry point for obtaining an OpenAI-compatible client
configured for the OpenCode Go endpoint.
"""
from __future__ import annotations

from openai import OpenAI

from winegpt.config import LLM_BASE_URL, OPENCODE_GO_API_KEY


def get_llm_client() -> OpenAI:
    """Return an OpenAI client configured for OpenCode Go."""
    if not OPENCODE_GO_API_KEY:
        raise ValueError("OPENCODE_GO_API_KEY not set in .env")
    return OpenAI(base_url=LLM_BASE_URL, api_key=OPENCODE_GO_API_KEY)
