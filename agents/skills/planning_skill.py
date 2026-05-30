from pydantic import BaseModel
from .llm_skill import llm_skill_structured


# Phase 13 — Planning Agent
# Analyzes topic + research and decides the best post type, angle, tone, and hook style
# BEFORE the writing agent starts. Writing agent receives this plan and writes accordingly.


class PostPlan(BaseModel):
    post_type: str   # "storytelling", "viral", "educational", "professional", "inspirational"
    angle: str       # specific angle to take (e.g. "Messi's discipline as a leadership lesson")
    tone: str        # "bold", "conversational", "data-driven", "emotional"
    hook_style: str  # "question", "bold_statement", "statistic", "story_opener"
    reason: str      # why this type was chosen for this topic


def planning_skill(topic: str, research: str) -> dict:
    """
    Analyzes topic and research, returns a structured post plan.
    Writing agent uses this plan to decide how to write — not just what to write.
    """
    result = llm_skill_structured(
        f"""You are a LinkedIn content strategist. Analyze this topic and research,
then decide the best content strategy.

Topic: {topic}
Research summary: {research}

Choose the best post_type:
- storytelling   : personal narrative, failure/success story, emotional journey
- viral          : bold opinion, controversial take, curiosity gap, pattern interrupt
- educational    : how-to, listicle, practical tips, step-by-step
- professional   : data-driven, industry insight, thought leadership, trend analysis
- inspirational  : achievement-based, motivational with real substance, not generic

Choose tone: bold / conversational / data-driven / emotional

Choose hook_style: question / bold_statement / statistic / story_opener

Respond with a JSON object with exactly these fields:
- "post_type": one of the 5 types above
- "angle": specific angle or story to tell (1 sentence, be specific)
- "tone": one of the 4 tones above
- "hook_style": one of the 4 hook styles above
- "reason": why you chose this type for this topic (1 sentence)""",
        PostPlan,
    )

    return {
        "post_type": result.post_type,
        "angle": result.angle,
        "tone": result.tone,
        "hook_style": result.hook_style,
        "reason": result.reason,
    }
