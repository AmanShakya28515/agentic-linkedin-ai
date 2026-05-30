from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from ..skills.review_skill import review_skill


# Phase 7 — Multi-Agent: Review Agent owns quality review responsibility
# Supervisor calls run_review_agent() and gets back {review_decision, review_feedback, final_post}

class ReviewAgentState(TypedDict):
    draft: str
    revision_count: int
    review_decision: str
    review_feedback: str
    final_post: str


def reviewer_node(state: ReviewAgentState):
    result = review_skill(draft=state["draft"], revision_count=state["revision_count"])
    if result["decision"] == "approved":
        return {
            "review_decision": "approved",
            "final_post": result["content"],
            "review_feedback": "",
        }
    return {
        "review_decision": "needs_work",
        "review_feedback": result["content"],
        "final_post": "",
    }


def build_review_agent():
    graph = StateGraph(ReviewAgentState)
    graph.add_node("reviewer", reviewer_node)
    graph.add_edge(START, "reviewer")
    graph.add_edge("reviewer", END)
    return graph.compile()


def run_review_agent(draft: str, revision_count: int) -> dict:
    agent = build_review_agent()
    result = agent.invoke({
        "draft": draft,
        "revision_count": revision_count,
        "review_decision": "",
        "review_feedback": "",
        "final_post": "",
    })
    return {
        "review_decision": result["review_decision"],
        "review_feedback": result["review_feedback"],
        "final_post": result["final_post"],
    }
