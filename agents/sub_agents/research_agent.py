import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from ..skills.search_tool import brave_search, get_news, get_reddit_discussions
from ..skills.hook_skill import hook_skill

load_dotenv()

# Phase 7 — manual graph: developer hardcoded research → hook flow
# class ResearchAgentState(TypedDict): ...
# def research_node(state): return {"research": research_skill(state["topic"])}
# def hook_node(state): return {"hook": hook_skill(state["research"])}
# def build_research_agent(): graph = StateGraph(...) ...
# def run_research_agent(topic): agent.invoke({"topic": topic, ...})

# Phase 10 — one tool (brave_search)
# Phase 14 — three tools: LLM decides which to use based on the topic
#   brave_search          → facts, stats, general info
#   get_news              → latest news and trending developments
#   get_reddit_discussions → real community opinions and experiences
llm = ChatOpenAI(
    model="meta-llama/llama-4-scout-17b-16e-instruct",
    base_url="https://api.groq.com/openai/v1",
    api_key=os.getenv("GROQ_API_KEY"),
)

react_agent = create_react_agent(llm, tools=[brave_search, get_news, get_reddit_discussions])


def run_research_agent(topic: str) -> dict:
    """
    Phase 10: LLM receives the topic + available tools.
    It decides to call search_web, reads results, may call again, then summarizes.
    Supervisor still calls run_research_agent() — interface unchanged.
    """
    result = react_agent.invoke({
        "messages": [("user",
            f"Research '{topic}' for a LinkedIn post. "
            f"You have 3 tools available — use the right ones for this topic:\n"
            f"- brave_search: facts, stats, company info, general knowledge\n"
            f"- get_news: recent news, trending events, latest developments\n"
            f"- get_reddit_discussions: real opinions, community experiences, debates\n\n"
            f"Use whichever tools give the richest research for this topic. "
            f"Then summarize the key findings in 150 words max."
        )]
    })

    # Extract the final text response from the last message
    research = result["messages"][-1].content

    # Hook generation still uses the same hook_skill
    hook = hook_skill(research)

    return {"research": research, "hook": hook}
