"""
LLM Generation  (n8n nodes: "LLM Generate Answer" + "Format Response")

Calls ChatOllama with the assembled prompt and strips chain-of-thought
<think> blocks produced by qwen3 / deepseek-r1 style models.
"""
from __future__ import annotations

import logging
import re

from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama

from config import settings

_THINK_RE = re.compile(r"<think>[\s\S]*?</think>", re.IGNORECASE)


def generate(prompt: str) -> str:
    """Invoke the LLM synchronously and return the cleaned answer text.

    Called from the async pipeline via ``asyncio.to_thread``.
    """
    kwargs = dict(
        model=settings.chat_model,
        base_url=settings.ollama_base_url,
        temperature=settings.chat_temperature,
        num_predict=settings.llm_num_predict,
    )
    if not settings.llm_think:
        # Disable qwen3 chain-of-thought via Ollama's options API
        kwargs["extra_body"] = {"think": False}
    llm = ChatOllama(**kwargs)
    response = llm.invoke([HumanMessage(content=prompt)])
    raw = response.content or ""
    logging.getLogger(__name__).debug("LLM raw output (%d chars): %r", len(raw), raw[:300])
    # Strip chain-of-thought reasoning blocks before returning
    cleaned = _THINK_RE.sub("", raw).strip()
    if not cleaned:
        logging.getLogger(__name__).warning("LLM returned only think content or empty; raw=%r", raw[:200])
    return cleaned
