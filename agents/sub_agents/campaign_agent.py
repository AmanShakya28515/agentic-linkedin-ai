from ..skills.campaign_skill import campaign_skill
from ..skills.research_skill import research_skill
from ..skills.hook_skill import hook_skill
from ..skills.writing_skill import writing_skill


# Phase 15 — Campaign Agent
# Generates a full multi-day content campaign.
# No HITL — generates all posts at once and returns them together.
#
# Flow:
#   1. campaign_skill → generates day-by-day plan
#   2. research_skill → researches the main topic once (shared across all days)
#   3. writing_skill  → writes each day's post using its specific plan
#   4. returns all posts as a list


def run_campaign_agent(topic: str, num_days: int = 5) -> dict:
    """
    Generates a complete content campaign.
    Returns {topic, posts: [{day, post_type, angle, content}]}
    """
    # Step 1: Generate campaign plan
    plan = campaign_skill(topic, num_days)

    # Step 2: Research topic once — shared context for all posts
    research = research_skill(topic)
    hook = hook_skill(research)

    # Step 3: Write each day's post using its specific plan
    posts = []
    for day_plan in plan:
        content = writing_skill(
            topic=f"{topic} — {day_plan['angle']}",
            research=research,
            hook=hook,
            plan={
                "post_type": day_plan["post_type"],
                "angle":     day_plan["angle"],
                "tone":      day_plan["tone"],
                "hook_style": "story_opener" if day_plan["post_type"] == "storytelling" else "bold_statement",
                "reason":    f"Day {day_plan['day']} of {num_days}-day campaign",
            },
        )
        posts.append({
            "day":       day_plan["day"],
            "post_type": day_plan["post_type"],
            "angle":     day_plan["angle"],
            "content":   content,
        })

    return {"topic": topic, "posts": posts}
