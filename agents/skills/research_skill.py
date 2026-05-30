from .llm_skill import llm_skill

# Phase 4 — research_skill asked LLM to research from training data (hallucinated facts)
# def research_skill(topic: str) -> str:
#     return llm_skill(f"Research this topic briefly for a LinkedIn post: {topic}")

# Phase 5 — Tool Use: search_tool fetches real web results first,
# then LLM summarizes actual data instead of making things up
# Phase 10 — renamed to search_web (@tool decorator), call .invoke() to run it
from .search_tool import brave_search


def research_skill(topic: str) -> str:
    """
    Searches the web for real, up-to-date information about the topic,
    then asks the LLM to summarize it for use in a LinkedIn post.
    """
    web_results = brave_search.invoke(topic)

    return llm_skill(
        f"""You are a research assistant preparing content for a LinkedIn post.

Topic: {topic}

Here are real web search results about this topic:
{web_results}

Summarize the key facts, stats, and insights from these results.
Focus on what is most relevant and credible for a professional LinkedIn audience.
Be concise — 150 words max."""
    )
