from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from ..skills.writing_skill import writing_skill


# Phase 7 — Writing Agent owns draft creation responsibility
# Phase 13 — Now also receives plan from Planning Agent

class WritingAgentState(TypedDict):
    topic: str
    research: str
    hook: str
    plan: dict           # Phase 13: post_type, angle, tone, hook_style
    review_feedback: str
    human_feedback: str
    draft: str


def writer_node(state: WritingAgentState):
    draft = writing_skill(
        topic=state["topic"],
        research=state["research"],
        hook=state["hook"],
        plan=state.get("plan", {}),
        review_feedback=state["review_feedback"],
        human_feedback=state["human_feedback"],
    )
    return {"draft": draft}


def build_writing_agent():
    graph = StateGraph(WritingAgentState)
    graph.add_node("writer", writer_node)
    graph.add_edge(START, "writer")
    graph.add_edge("writer", END)
    return graph.compile()


def run_writing_agent(topic: str, research: str, hook: str, plan: dict = {},
                      review_feedback: str = "", human_feedback: str = "") -> dict:
    agent = build_writing_agent()
    result = agent.invoke({
        "topic": topic,
        "research": research,
        "hook": hook,
        "plan": plan,
        "review_feedback": review_feedback,
        "human_feedback": human_feedback,
        "draft": "",
    })
    return {"draft": result["draft"]}
