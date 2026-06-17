from .llm_skill import llm_skill, llm_skill_stream

# Phase 8 and before — no memory, prompt had no user preferences
# def writing_skill(topic, research, hook, review_feedback="", human_feedback=""):
#     ...
#     return llm_skill(f"Write a LinkedIn post.\nTopic: {topic}\n...")

# Phase 9 — Long-Term Memory: INJECT user preferences into prompt before writing
from .memory_skill import get_memory_context
from .rag_skill import retrieve_similar_posts        # Phase 11
from .prompt_enhancer_skill import prompt_enhancer_skill  # Phase 12
from .persona_skill import get_persona_prompt        # Phase 18


def _build_prompt(topic, research, hook, feedback_section, memory_context, human_feedback, rag_context="", plan=None, persona="default"):
    if memory_context and human_feedback:
        memory_section = f"\nSTRICT USER RULES (must follow — no exceptions):\n{memory_context}"
        override_note = "\nCURRENT INSTRUCTION (overrides rules for THIS draft only): apply the human feedback above."
    elif memory_context:
        memory_section = f"\nSTRICT USER RULES (must follow — no exceptions):\n{memory_context}"
        override_note = ""
    else:
        memory_section = ""
        override_note = ""

    rag_section = f"\nCRITICAL — USE THIS INFORMATION (from uploaded documents/knowledge base). You MUST reference specific facts, numbers, and details from this content in your post:\n{rag_context}" if rag_context else ""

    # Phase 18 — inject persona voice profile
    persona_section = get_persona_prompt(persona)

    # Phase 13 — inject plan from Planning Agent
    if plan:
        plan_section = f"""
CONTENT PLAN (follow this strategy):
- Post type: {plan.get('post_type', '')}
- Angle: {plan.get('angle', '')}
- Tone: {plan.get('tone', '')}
- Hook style: {plan.get('hook_style', '')}
- Why this approach: {plan.get('reason', '')}
"""
    else:
        plan_section = ""

    return f"""You are a professional LinkedIn content creator known for bold, punchy storytelling.
{persona_section}
Writing brief: {topic}
{plan_section}

Research: {research}
Hooks: {hook}
{feedback_section}
{memory_section}
{override_note}
{rag_section}

STRICT LINKEDIN CREATOR RULES (no exceptions):

STYLE:
- Write like a top LinkedIn creator, NOT a textbook or Wikipedia
- Use emotional tension — make the reader feel something
- Use curiosity-driven hooks — first line must make them stop scrolling
- Prefer storytelling over explanation
- Use contrast and bold opinions ("Most people think X. They're wrong.")
- Create mobile-friendly formatting — short lines, white space, easy to skim
- Make every line pull the reader to the next line

AVOID:
- Generic motivational language ("believe in yourself", "hard work pays off")
- Essay-style paragraphs
- Phrases like "widely regarded", "testament to", "it is worth noting", "in conclusion"
- Textbook summaries
- Passive voice

FORMAT:
- Start with a single punchy hook line
- Max 1-2 sentences per paragraph
- Use line breaks between every paragraph
- End with a question or a call to action
- Keep under 250 words"""


def writing_skill(topic: str, research: str, hook: str, plan: dict = {},
                  review_feedback: str = "", human_feedback: str = "",
                  persona: str = "default") -> str:
    feedback_section = ""
    if review_feedback:
        feedback_section += f"\nReviewer feedback to address:\n{review_feedback}"
    if human_feedback:
        feedback_section += f"\nHuman feedback to address:\n{human_feedback}"

    enhanced_topic = prompt_enhancer_skill(topic) if not human_feedback else topic
    memory_context = get_memory_context()
    rag_context = retrieve_similar_posts(topic)
    prompt = _build_prompt(enhanced_topic, research, hook, feedback_section, memory_context, human_feedback, rag_context, plan, persona)
    return llm_skill(prompt)


# Phase 6 — Streaming (Phase 9/11/12/13/18 updates)
def writing_skill_stream(topic: str, research: str, hook: str, plan: dict = {},
                         review_feedback: str = "", human_feedback: str = "",
                         persona: str = "default"):
    feedback_section = ""
    if review_feedback:
        feedback_section += f"\nReviewer feedback to address:\n{review_feedback}"
    if human_feedback:
        feedback_section += f"\nHuman feedback to address:\n{human_feedback}"

    enhanced_topic = prompt_enhancer_skill(topic) if not human_feedback else topic
    memory_context = get_memory_context()
    rag_context = retrieve_similar_posts(topic)
    prompt = _build_prompt(enhanced_topic, research, hook, feedback_section, memory_context, human_feedback, rag_context, plan, persona)
    yield from llm_skill_stream(prompt)
