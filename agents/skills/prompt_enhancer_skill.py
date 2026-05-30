from .llm_skill import llm_skill


def prompt_enhancer_skill(topic: str) -> str:
    """
    Expands a bare topic into a rich LinkedIn writing brief.
    "messi" → "Write a bold post about Messi's discipline and consistency,
                using short punchy lines and leadership lessons for professionals."
    """
    return llm_skill(
        f"""A user wants a LinkedIn post about: "{topic}"

Expand this into a specific, vivid writing brief. Include:
- The angle or story to tell (leadership, failure, discipline, innovation, etc.)
- The target emotion (inspiring, provocative, surprising, practical)
- The audience (professionals, entrepreneurs, job seekers, etc.)
- Suggested tone (bold, conversational, storytelling, data-driven)

Keep it under 3 sentences. Be specific, not generic.
Return ONLY the enhanced writing brief. No labels, no explanation."""
    )
