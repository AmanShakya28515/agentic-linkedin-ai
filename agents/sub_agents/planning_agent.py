from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from ..skills.planning_skill import planning_skill


# Phase 13 — Planning Agent
# Owns the content strategy decision.
# Supervisor calls run_planning_agent() and gets back a structured post plan.
# Writing agent then receives this plan and writes accordingly.

class PlanningAgentState(TypedDict):
    topic: str
    research: str
    plan: dict


def planning_node(state: PlanningAgentState):
    plan = planning_skill(state["topic"], state["research"])
    return {"plan": plan}


def build_planning_agent():
    graph = StateGraph(PlanningAgentState)
    graph.add_node("planner", planning_node)
    graph.add_edge(START, "planner")
    graph.add_edge("planner", END)
    return graph.compile()


def run_planning_agent(topic: str, research: str) -> dict:
    """
    Supervisor calls this. Returns {plan} with post_type, angle, tone, hook_style.
    Supervisor doesn't know HOW the plan is made — just receives the result.
    """
    agent = build_planning_agent()
    result = agent.invoke({"topic": topic, "research": research, "plan": {}})
    return {"plan": result["plan"]}
