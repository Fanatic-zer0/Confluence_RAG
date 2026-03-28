"""
Prompt Assembly  (n8n node: "Prompt Assembly")

Builds the final LLM prompt from retrieved chunks and extracts
the sources list for citation in the response.
"""
from __future__ import annotations

from typing import List, Tuple

from models import Chunk, Source

_SYSTEM_PROMPT = """\
You are an internal Confluence assistant for our organization.
Answer concisely and factually using ONLY the provided context.
If the answer is not in the context, say you don't know and suggest \
where it might be documented.
Always include a "Sources" section listing the URLs you used.\
"""


def assemble(
    chunks: List[Chunk],
    question: str,
) -> Tuple[str, List[Source]]:
    """Build the final LLM prompt and return it together with the sources list.

    Returns:
        (prompt_text, sources)
    """
    context_blocks: List[str] = []
    sources: List[Source] = []

    for idx, c in enumerate(chunks, start=1):
        context_blocks.append(
            f"Source {idx} (score={c.score:.3f}):\n"
            f"Title: {c.title}\n"
            f"URL: {c.source_url}\n"
            f"Content:\n{c.chunk_text}"
        )
        sources.append(Source(
            label=f"Source {idx}",
            title=c.title,
            url=c.source_url,
            score=c.score,
        ))

    context_text = "\n\n------------------------\n\n".join(context_blocks)

    prompt = (
        f"{_SYSTEM_PROMPT}\n\n"
        f"======== CONTEXT START ========\n"
        f"{context_text}\n"
        f"======== CONTEXT END =========\n\n"
        f"User question:\n{question}\n\n"
        f"Answer in markdown. Use bullet points where useful."
    )

    return prompt, sources
