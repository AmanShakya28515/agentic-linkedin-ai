from pydantic import BaseModel
from typing import List
from .llm_skill import llm_skill_structured


# Phase 15 — Multi-Post Campaign
# Generates a day-by-day content plan for a campaign topic.
# Each day has a different post type, angle, and tone — building a cohesive narrative.


class DayPlan(BaseModel):
    day: int
    post_type: str  # storytelling, viral, educational, professional, inspirational
    angle: str      # specific angle for this day
    tone: str       # bold, conversational, data-driven, emotional


class CampaignPlan(BaseModel):
    days: List[DayPlan]


def campaign_skill(topic: str, num_days: int = 5) -> list:
    """
    Generates a structured day-by-day campaign plan.
    Returns a list of DayPlan dicts — one per day.
    """
    result = llm_skill_structured(
        f"""Create a {num_days}-day LinkedIn content campaign about "{topic}".

Each day should be DIFFERENT — vary the post type, angle, and tone to build a cohesive campaign.

Suggested flow:
- Day 1: Grab attention (viral or bold hook)
- Day 2: Educate (tips, how-to, insights)
- Day 3: Tell a story (personal narrative or case study)
- Day 4: Share data (stats, trends, numbers)
- Day 5: Call to action (what the audience should do now)

For each day return:
- "day": day number
- "post_type": one of: storytelling, viral, educational, professional, inspirational
- "angle": specific angle or story for this day (1 sentence, be specific)
- "tone": one of: bold, conversational, data-driven, emotional

Respond with a JSON object with a "days" array containing {num_days} day objects.""",
        CampaignPlan,
    )

    return [
        {
            "day": d.day,
            "post_type": d.post_type,
            "angle": d.angle,
            "tone": d.tone,
        }
        for d in result.days
    ]
