import json
import os
from .llm_skill import llm_skill

# Phase 9 — Long-Term Memory
# Stores user preferences and past topics in a JSON file that persists across sessions.
# 4 steps: EXTRACT → STORE → RETRIEVE → INJECT (into writing prompt)

# Phase 23 — Railway: use DATA_DIR env var so user_memory.json persists on Railway volume
_BASE = os.environ.get('DATA_DIR', os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
MEMORY_FILE = os.path.join(_BASE, "user_memory.json")


def read_memory() -> dict:
    if not os.path.exists(MEMORY_FILE):
        return {"preferences": [], "past_topics": []}
    with open(MEMORY_FILE, "r") as f:
        return json.load(f)


def write_memory(memory: dict):
    with open(MEMORY_FILE, "w") as f:
        json.dump(memory, f, indent=2)


def is_persistent_preference(feedback: str) -> bool:
    """
    Classifies feedback before saving.
    One-time instructions ("make this one longer") → NOT saved.
    Persistent preferences ("I always prefer short posts") → saved.
    """
    response = llm_skill(
        f"""A user gave this feedback on a LinkedIn post: "{feedback}"

Is this a PERSISTENT preference (should apply to all future posts)?
Or a ONE-TIME instruction (applies only to this specific post)?

Persistent examples:
- "I always like short posts"
- "never use hashtags"
- "from now on be more casual"
- "I prefer bullet points"

One-time examples:
- "make this one longer"
- "add emojis to this post"
- "make this more formal for this one"
- "can you expand this specific section"

Reply with ONLY one word: persistent OR one-time"""
    )
    return "persistent" in response.strip().lower()


def save_preferences(feedback: str):
    """
    Phase 9 original: saved every feedback immediately — wrong.
    Phase 9 fix: classify first. Only persistent preferences get saved.
    One-time instructions are applied to current draft but NOT stored in memory.
    """
    if not is_persistent_preference(feedback):
        return  # one-time instruction — skip saving

    response = llm_skill(
        f"""A user gave this feedback on a LinkedIn post draft: "{feedback}"

Extract the writing preference as a short phrase (under 8 words).
Examples: "prefers short posts", "no hashtags", "professional tone", "wants bullet points"

If no clear preference, return exactly: none
Reply with ONLY the preference phrase or "none"."""
    )

    preference = response.strip().lower()
    if preference != "none" and preference:
        memory = read_memory()

        # Remove contradicting preferences before adding new one
        # e.g. if memory has "no tags" and new pref is "wants tags" → remove "no tags"
        opposite = llm_skill(
            f"""Current memory preferences: {memory["preferences"]}
New preference to add: "{preference}"

List any existing preferences that directly CONTRADICT the new one.
Return them as a comma-separated list, or "none" if no contradictions.
Reply with ONLY the list or "none"."""
        )
        if opposite.strip().lower() != "none":
            contradictions = [c.strip().lower() for c in opposite.split(",")]
            memory["preferences"] = [
                p for p in memory["preferences"]
                if p.lower() not in contradictions
            ]

        if preference not in memory["preferences"]:
            memory["preferences"].append(preference)
        write_memory(memory)


def save_topic(topic: str):
    """STORE: save a completed topic so agent knows what's been covered."""
    memory = read_memory()
    if topic not in memory["past_topics"]:
        memory["past_topics"].append(topic)
        memory["past_topics"] = memory["past_topics"][-20:]  # keep last 20
    write_memory(memory)


def get_memory_context() -> str:
    """RETRIEVE: read memory and format as context string for the writing prompt."""
    memory = read_memory()
    parts = []

    if memory["preferences"]:
        parts.append("User preferences (always apply these): " + ", ".join(memory["preferences"]))
    if memory["past_topics"]:
        parts.append("Past topics already covered: " + ", ".join(memory["past_topics"][-5:]))

    return "\n".join(parts) if parts else ""
