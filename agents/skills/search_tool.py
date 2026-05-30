from duckduckgo_search import DDGS
from langchain_core.tools import tool


# Phase 5 — plain function, called directly by research_skill (developer decides when to call)
# def search_tool(query: str, max_results: int = 5) -> str:
#     ...


# Phase 10 — Formal Tool Use: @tool decorator exposes this to the LLM
# LLM now sees the function signature + docstring and decides when/if/how many times to call it
# search_web → renamed to brave_search
# Llama models on Groq are fine-tuned to call "brave_search" by default
# Our implementation still uses DuckDuckGo — the name is just what the LLM expects
@tool
def brave_search(query: str) -> str:
    """Search the web for real-time information about a topic.
    Use this to find current facts, statistics, and news."""
    results = []

    with DDGS() as ddgs:
        for r in ddgs.text(query, max_results=5):
            title = r.get("title", "")
            body  = r.get("body", "")
            href  = r.get("href", "")
            results.append(f"- {title}: {body} ({href})")

    if not results:
        return f"No search results found for: {query}"

    return "\n".join(results)


# Phase 14 — Multiple Tools: LLM now decides WHICH tool to use based on the topic

@tool
def get_news(query: str) -> str:
    """Search for the latest news articles about a topic.
    Use this when you need recent events, breaking news, or trending developments."""
    try:
        results = []
        with DDGS() as ddgs:
            for r in ddgs.news(query, max_results=5):
                title  = r.get("title", "")
                body   = r.get("body", "")
                date   = r.get("date", "")
                source = r.get("source", "")
                results.append(f"- [{date}] {source}: {title} — {body}")

        if results:
            return "\n".join(results)

    except Exception:
        pass

    # Fallback: use web search with "latest news" if news endpoint is rate limited
    fallback = []
    with DDGS() as ddgs:
        for r in ddgs.text(f"{query} latest news 2025", max_results=5):
            title = r.get("title", "")
            body  = r.get("body", "")
            fallback.append(f"- {title}: {body}")

    return "\n".join(fallback) if fallback else f"No news found for: {query}"


@tool
def get_reddit_discussions(query: str) -> str:
    """Search Reddit for real community discussions, opinions and experiences about a topic.
    Use this when you need authentic human perspectives, debates, or real-world stories."""
    results = []

    with DDGS() as ddgs:
        for r in ddgs.text(f"site:reddit.com {query}", max_results=5):
            title = r.get("title", "")
            body  = r.get("body", "")
            href  = r.get("href", "")
            results.append(f"- {title}: {body} ({href})")

    if not results:
        return f"No Reddit discussions found for: {query}"

    return "\n".join(results)
