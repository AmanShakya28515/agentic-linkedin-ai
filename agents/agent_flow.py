from typing import TypedDict
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver  # Added in Phase 2 - HITL
from langgraph.types import interrupt, Command        # Added in Phase 2 - HITL

# Phase 3 - Skills: import reusable skills
# Nodes no longer call LLM directly — they call skills instead
from .skills.research_skill import research_skill
from .skills.hook_skill import hook_skill
from .skills.writing_skill import writing_skill
from .skills.review_skill import review_skill

load_dotenv()

# Phase 2 original: OpenAI client lived here
# Phase 3: client moved into llm_skill.py — agent_flow.py no longer knows about the LLM
# Phase 1 original: client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
# Phase 2: client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=...)

MAX_REVISIONS = 2  # Added in Phase 2 - Conditional Routing

# Added in Phase 2 - HITL: persists graph state across requests so workflow can be paused and resumed
memory = MemorySaver()


# =============================================================================
# PHASE 1 - Original State
# =============================================================================
# class AgentState(TypedDict):
#     topic: str
#     research: str
#     hook: str
#     draft: str
#     final_post: str

# =============================================================================
# PHASE 2 - Conditional Routing: added review_decision, review_feedback, revision_count
# PHASE 2 - HITL: added human_feedback, human_approved
# =============================================================================
class AgentState(TypedDict):
    topic: str
    research: str
    hook: str
    draft: str
    review_decision: str    # Phase 2 Conditional Routing: "approved" or "needs_work"
    review_feedback: str    # Phase 2 Conditional Routing: reviewer's feedback to writer
    revision_count: int     # Phase 2 Conditional Routing: tracks how many times writer revised
    human_feedback: str     # Phase 2 HITL: feedback from human reviewer
    human_approved: bool    # Phase 2 HITL: whether human approved the draft
    final_post: str


# Phase 1 & 2 — ask_gemini lived here, nodes called it directly
# Phase 3 — moved into skills/llm_skill.py, nodes now call individual skills
# def ask_gemini(prompt: str) -> str:
#     response = client.chat.completions.create(
#         model="nvidia/nemotron-3-nano-30b-a3b:free",
#         messages=[{"role": "user", "content": prompt}],
#     )
#     return response.choices[0].message.content


# =============================================================================
# NODES
# =============================================================================

# Phase 1 & 2 research_node:
# def research_node(state: AgentState):
#     research = ask_gemini(f"Research this topic briefly for a LinkedIn post: {state['topic']}")
#     return {"research": research}

# Phase 3 research_node — calls research_skill instead of ask_gemini directly
def research_node(state: AgentState):
    return {"research": research_skill(state["topic"])}


# Phase 1 & 2 hook_node:
# def hook_node(state: AgentState):
#     hook = ask_gemini(f"Create 3 strong LinkedIn hooks based on this research:\n{state['research']}")
#     return {"hook": hook}

# Phase 3 hook_node — calls hook_skill
def hook_node(state: AgentState):
    return {"hook": hook_skill(state["research"])}


# =============================================================================
# PHASE 1 - writer_node (no feedback, just writes)
# =============================================================================
# def writer_node(state: AgentState):
#     draft = ask_gemini(
#         f"""
# Write a LinkedIn post.
# Topic: {state['topic']}
# Research: {state['research']}
# Hooks: {state['hook']}
# Make it simple, clear, and professional.
# """
#     )
#     return {"draft": draft}

# =============================================================================
# PHASE 2 - writer_node (used ask_gemini directly with feedback)
# =============================================================================
# def writer_node(state: AgentState):
#     feedback_section = ""
#     if state["review_feedback"]:
#         feedback_section = f"\nReviewer feedback to address:\n{state['review_feedback']}"
#     if state["human_feedback"]:
#         feedback_section += f"\nHuman feedback to address:\n{state['human_feedback']}"
#     draft = ask_gemini(f"""Write a LinkedIn post...""")
#     return {"draft": draft}

# Phase 3 - writer_node calls writing_skill
def writer_node(state: AgentState):
    draft = writing_skill(
        topic=state["topic"],
        research=state["research"],
        hook=state["hook"],
        review_feedback=state["review_feedback"],
        human_feedback=state["human_feedback"],
    )
    return {"draft": draft}


# =============================================================================
# PHASE 2 - HITL: new node — graph pauses here and waits for human input
# =============================================================================
# Phase 2 original (routed to reviewer regardless of human decision):
# def human_review_node(state: AgentState):
#     human_input = interrupt({...})
#     return {
#         "human_feedback": human_input.get("feedback", ""),
#         "human_approved": human_input.get("approved", True),
#     }

# Phase 2 fix: human approval = final authority
# If human approves → set final_post = draft, route to END (skip AI reviewer)
# If human rejects → send feedback to writer for rewrite
def human_review_node(state: AgentState):
    # interrupt() saves the graph state and returns control to the API caller
    # When /resume/ is called, execution continues from this exact line
    human_input = interrupt({
        "draft": state["draft"],
        "message": "Review this draft. Respond with approved (true/false) and optional feedback.",
    })

    approved = human_input.get("approved", True)

    if approved:
        return {
            "human_approved": True,
            "human_feedback": "",
            "final_post": state["draft"],   # draft becomes final post immediately
        }

    return {
        "human_approved": False,
        "human_feedback": human_input.get("feedback", ""),
    }


# =============================================================================
# PHASE 1 - reviewer_node (just rewrites, no decision)
# =============================================================================
# def reviewer_node(state: AgentState):
#     final_post = ask_gemini(
#         f"""
# Review and improve this LinkedIn post.
# Make it clearer, shorter, and more engaging.
# Draft:
# {state['draft']}
# """
#     )
#     return {"final_post": final_post}

# =============================================================================
# PHASE 2 - reviewer_node (used ask_gemini directly with full prompt + parsing)
# =============================================================================
# def reviewer_node(state: AgentState):
#     response = ask_gemini(f"""You are a strict reviewer... DECISION: ...""")
#     decision = "approved"
#     for line in response.strip().split("\n"):
#         if line.startswith("DECISION:"):
#             decision = line.replace("DECISION:", "").strip().lower()
#             break
#     if decision == "approved":
#         ...
#     else:
#         ...

# Phase 3 - reviewer_node calls review_skill (prompt + parsing moved into skill)
def reviewer_node(state: AgentState):
    result = review_skill(
        draft=state["draft"],
        revision_count=state["revision_count"],
    )

    if result["decision"] == "approved":
        return {
            "review_decision": "approved",
            "final_post": result["content"],
            "review_feedback": "",
        }
    else:
        return {
            "review_decision": "needs_work",
            "review_feedback": result["content"],
            "revision_count": state["revision_count"] + 1,
        }


# =============================================================================
# PHASE 2 - Conditional Routing: routes graph after reviewer decision
# =============================================================================
def route_after_human(state: AgentState) -> str:
    # Phase 2 original: human approved → reviewer (AI could still override)
    # Phase 2 fix: human approved → END (human is final authority)
    # Human rejected → writer (rewrite with feedback)
    if state["human_approved"]:
        return "end"
    return "writer"


def route_after_review(state: AgentState) -> str:
    if state["review_decision"] == "approved" or state["revision_count"] >= MAX_REVISIONS:
        return "end"
    return "writer"


# =============================================================================
# PHASE 6 - Streaming: routes START → research (normal) or human_review (draft pre-provided)
# =============================================================================
def route_from_start(state: AgentState) -> str:
    if state.get("draft"):
        return "human_review"
    return "research"


# =============================================================================
# PHASE 1 - build_graph (straight sequential edges, no branching)
# =============================================================================
# def build_graph():
#     graph = StateGraph(AgentState)
#     graph.add_node("research", research_node)
#     graph.add_node("hook", hook_node)
#     graph.add_node("writer", writer_node)
#     graph.add_node("reviewer", reviewer_node)
#     graph.add_edge(START, "research")
#     graph.add_edge("research", "hook")
#     graph.add_edge("hook", "writer")
#     graph.add_edge("writer", "reviewer")
#     graph.add_edge("reviewer", END)
#     return graph.compile()

# =============================================================================
# PHASE 2 - Conditional Routing only (no HITL, no human_review node)
# =============================================================================
# def build_graph():
#     graph = StateGraph(AgentState)
#     graph.add_node("research", research_node)
#     graph.add_node("hook", hook_node)
#     graph.add_node("writer", writer_node)
#     graph.add_node("reviewer", reviewer_node)
#     graph.add_edge(START, "research")
#     graph.add_edge("research", "hook")
#     graph.add_edge("hook", "writer")
#     graph.add_edge("writer", "reviewer")
#     graph.add_conditional_edges(
#         "reviewer",
#         route_after_review,
#         {"writer": "writer", "end": END}
#     )
#     return graph.compile()

# =============================================================================
# PHASE 2 - HITL: adds human_review node between writer and reviewer
# =============================================================================
def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("research", research_node)
    graph.add_node("hook", hook_node)
    graph.add_node("writer", writer_node)
    graph.add_node("human_review", human_review_node)   # Phase 2 HITL
    graph.add_node("reviewer", reviewer_node)

    # Phase 5 and before:
    # graph.add_edge(START, "research")

    # Phase 6 — Streaming: if draft already provided (pre-streamed), skip to human_review
    graph.add_conditional_edges(START, route_from_start, {
        "research": "research",
        "human_review": "human_review",
    })
    graph.add_edge("research", "hook")
    graph.add_edge("hook", "writer")
    graph.add_edge("writer", "human_review")            # Phase 2 HITL
    graph.add_conditional_edges(                        # Phase 2 HITL fix
        "human_review",
        route_after_human,
        {"writer": "writer", "reviewer": "reviewer", "end": END}
    )
    graph.add_conditional_edges(
        "reviewer",
        route_after_review,
        {"writer": "writer", "end": END}
    )

    return graph.compile(checkpointer=memory)           # Phase 2 HITL: needs checkpointer


# =============================================================================
# PHASE 1 - run_agent_flow (simple invoke, no config needed)
# =============================================================================
# def run_agent_flow(topic: str):
#     app = build_graph()
#     initial_state = {
#         "topic": topic,
#         "research": "",
#         "hook": "",
#         "draft": "",
#         "final_post": "",
#     }
#     return app.invoke(initial_state)

# =============================================================================
# PHASE 2 - run_agent_flow (needs config for checkpointer)
# =============================================================================
def run_agent_flow(topic: str):
    app = build_graph()
    config = {"configurable": {"thread_id": "default"}}

    initial_state = {
        "topic": topic,
        "research": "",
        "hook": "",
        "draft": "",
        "review_decision": "",
        "review_feedback": "",
        "revision_count": 0,
        "human_feedback": "",
        "human_approved": False,
        "final_post": "",
    }

    return app.invoke(initial_state, config=config)


# =============================================================================
# PHASE 1 - stream_agent_flow (simple stream, no config)
# =============================================================================
# def stream_agent_flow(topic: str) -> list:
#     app = build_graph()
#     initial_state = {
#         "topic": topic,
#         "research": "",
#         "hook": "",
#         "draft": "",
#         "final_post": "",
#     }
#     steps = []
#     for step in app.stream(initial_state):
#         for node_name, state_update in step.items():
#             steps.append({"node": node_name, "state_update": state_update})
#     return steps

# =============================================================================
# PHASE 2 - stream_agent_flow (needs config for checkpointer)
# =============================================================================
def stream_agent_flow(topic: str) -> list:
    app = build_graph()
    config = {"configurable": {"thread_id": "stream-default"}}

    initial_state = {
        "topic": topic,
        "research": "",
        "hook": "",
        "draft": "",
        "review_decision": "",
        "review_feedback": "",
        "revision_count": 0,
        "human_feedback": "",
        "human_approved": False,
        "final_post": "",
    }

    steps = []
    for step in app.stream(initial_state, config=config):
        for node_name, state_update in step.items():
            steps.append({"node": node_name, "state_update": state_update})

    return steps


# =============================================================================
# PHASE 2 - HITL: new functions for start/resume workflow
# =============================================================================
def start_agent_flow(topic: str, thread_id: str) -> dict:
    app = build_graph()
    config = {"configurable": {"thread_id": thread_id}}

    initial_state = {
        "topic": topic,
        "research": "",
        "hook": "",
        "draft": "",
        "review_decision": "",
        "review_feedback": "",
        "revision_count": 0,
        "human_feedback": "",
        "human_approved": False,
        "final_post": "",
    }

    # Runs until interrupt() is hit — graph pauses and returns state at that point
    result = app.invoke(initial_state, config=config)
    return result


def resume_agent_flow(thread_id: str, human_input: dict) -> dict:
    app = build_graph()
    config = {"configurable": {"thread_id": thread_id}}

    # Resumes from checkpoint — passes human_input back to interrupt()
    result = app.invoke(Command(resume=human_input), config=config)
    return result


# =============================================================================
# PHASE 6 - Streaming: starts workflow at human_review with a pre-streamed draft
# Skips research/hook/writer — those already ran during streaming
# =============================================================================
def start_agent_flow_with_draft(topic: str, research: str, hook: str,
                                 draft: str, thread_id: str) -> dict:
    app = build_graph()
    config = {"configurable": {"thread_id": thread_id}}

    initial_state = {
        "topic": topic,
        "research": research,
        "hook": hook,
        "draft": draft,          # pre-provided → route_from_start sends to human_review
        "review_decision": "",
        "review_feedback": "",
        "revision_count": 0,
        "human_feedback": "",
        "human_approved": False,
        "final_post": "",
    }

    result = app.invoke(initial_state, config=config)
    return result
