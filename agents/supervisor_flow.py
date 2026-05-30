from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command

from .sub_agents.research_agent import run_research_agent
from .sub_agents.writing_agent import run_writing_agent
from .sub_agents.review_agent import run_review_agent
from .sub_agents.planning_agent import run_planning_agent      # Phase 13
from .skills.memory_skill import save_preferences, save_topic  # Phase 9
from .skills.rag_skill import store_post                       # Phase 11

# =============================================================================
# PHASE 7 — Multi-Agent (Supervisor Pattern)
#
# Before (Phase 1-6): one monolithic graph owned all nodes directly
# After  (Phase 7):   supervisor delegates to specialized sub-agents
#
# Supervisor only knows the INTERFACE of each sub-agent, not the internals:
#   run_research_agent(topic)              → {research, hook}
#   run_writing_agent(topic, research, ...) → {draft}
#   run_review_agent(draft, revision_count) → {review_decision, ...}
#
# HITL (human_review) stays in supervisor — it owns workflow coordination
# =============================================================================

MAX_REVISIONS = 2
memory = MemorySaver()


class SupervisorState(TypedDict):
    topic: str
    research: str
    hook: str
    plan: dict           # Phase 13 — planning agent decides post type + angle
    draft: str
    review_decision: str
    review_feedback: str
    revision_count: int
    human_feedback: str
    human_approved: bool
    final_post: str


# ── Supervisor nodes — each delegates to a sub-agent ──────────────

def research_agent_node(state: SupervisorState):
    result = run_research_agent(state["topic"])
    return {"research": result["research"], "hook": result["hook"]}


# Phase 13 — Planning Agent: decides post type before writing
def planning_agent_node(state: SupervisorState):
    result = run_planning_agent(state["topic"], state["research"])
    return {"plan": result["plan"]}


def writing_agent_node(state: SupervisorState):
    result = run_writing_agent(
        topic=state["topic"],
        research=state["research"],
        hook=state["hook"],
        plan=state.get("plan", {}),        # Phase 13: pass plan to writing agent
        review_feedback=state["review_feedback"],
        human_feedback=state["human_feedback"],
    )
    return {"draft": result["draft"]}


def human_review_node(state: SupervisorState):
    human_input = interrupt({
        "draft": state["draft"],
        "message": "Review this draft. Respond with approved (true/false) and optional feedback.",
    })
    approved = human_input.get("approved", True)

    if approved:
        # Phase 9 — STORE: save topic so agent knows it's been covered
        save_topic(state["topic"])
        # Phase 11 — STORE: save approved post to vector database for RAG
        store_post(topic=state["topic"], post=state["draft"])
        return {"human_approved": True, "human_feedback": "", "final_post": state["draft"]}

    feedback = human_input.get("feedback", "")
    # Phase 9 — EXTRACT + STORE: pull preferences from feedback and save them
    if feedback:
        save_preferences(feedback)
    return {"human_approved": False, "human_feedback": feedback}


def review_agent_node(state: SupervisorState):
    result = run_review_agent(
        draft=state["draft"],
        revision_count=state["revision_count"],
    )
    if result["review_decision"] == "approved":
        return {
            "review_decision": "approved",
            "final_post": result["final_post"],
            "review_feedback": "",
        }
    return {
        "review_decision": "needs_work",
        "review_feedback": result["review_feedback"],
        "revision_count": state["revision_count"] + 1,
    }


# ── Routing ────────────────────────────────────────────────────────

def route_after_human(state: SupervisorState) -> str:
    if state["human_approved"]:
        return "end"
    return "writing_agent"


def route_after_review(state: SupervisorState) -> str:
    if state["review_decision"] == "approved" or state["revision_count"] >= MAX_REVISIONS:
        return "end"
    return "writing_agent"


# ── Graph ──────────────────────────────────────────────────────────

# Phase 6+7: if draft pre-provided (streamed), skip research/writing and go to human_review
def route_from_start(state: SupervisorState) -> str:
    if state.get("draft"):
        return "human_review"
    return "research_agent"


def build_supervisor():
    graph = StateGraph(SupervisorState)

    graph.add_node("research_agent", research_agent_node)
    graph.add_node("planning_agent", planning_agent_node)  # Phase 13
    graph.add_node("writing_agent", writing_agent_node)
    graph.add_node("human_review", human_review_node)
    graph.add_node("review_agent", review_agent_node)

    # Phase 6+7 fix: conditional start — skip to human_review if draft already streamed
    graph.add_conditional_edges(START, route_from_start, {
        "research_agent": "research_agent",
        "human_review": "human_review",
    })
    graph.add_edge("research_agent", "planning_agent")   # Phase 13: research → plan → write
    graph.add_edge("planning_agent", "writing_agent")
    graph.add_edge("writing_agent", "human_review")
    graph.add_conditional_edges("human_review", route_after_human, {
        "writing_agent": "writing_agent",
        "end": END,
    })
    graph.add_conditional_edges("review_agent", route_after_review, {
        "writing_agent": "writing_agent",
        "end": END,
    })

    return graph.compile(checkpointer=memory)


# ── Entry points (mirror of agent_flow.py) ────────────────────────

def start_supervisor_flow(topic: str, thread_id: str) -> dict:
    app = build_supervisor()
    config = {"configurable": {"thread_id": thread_id}}
    initial_state = {
        "topic": topic,
        "research": "",
        "hook": "",
        "plan": {},
        "draft": "",
        "review_decision": "",
        "review_feedback": "",
        "revision_count": 0,
        "human_feedback": "",
        "human_approved": False,
        "final_post": "",
    }
    result = app.invoke(initial_state, config=config)
    return result


def resume_supervisor_flow(thread_id: str, human_input: dict) -> dict:
    app = build_supervisor()
    config = {"configurable": {"thread_id": thread_id}}
    result = app.invoke(Command(resume=human_input), config=config)
    return result


# Phase 6+7: streaming endpoint pre-generates the draft, then saves it here
def start_supervisor_flow_with_draft(topic: str, research: str, hook: str,
                                      draft: str, thread_id: str) -> dict:
    app = build_supervisor()
    config = {"configurable": {"thread_id": thread_id}}
    initial_state = {
        "topic": topic,
        "research": research,
        "hook": hook,
        "plan": {},
        "draft": draft,
        "review_decision": "",
        "review_feedback": "",
        "revision_count": 0,
        "human_feedback": "",
        "human_approved": False,
        "final_post": "",
    }
    result = app.invoke(initial_state, config=config)
    return result
