from pydantic import BaseModel
from .llm_skill import llm_skill_structured


# Phase 4-7 — manual text parsing (fragile: what if LLM adds punctuation or extra words?)
# def orchestrator_skill(user_message, current_state):
#     response = llm_skill("...Reply with ONLY one word...")
#     intent = response.strip().lower().split()[0] if response.strip() else "feedback"
#     if intent not in ("new_topic", "feedback", "approved"):
#         intent = "feedback"
#     return intent


# Phase 8 — Structured Output: Pydantic guarantees intent is always a clean typed field
class IntentResult(BaseModel):
    intent: str   # "new_topic", "feedback", or "approved"


def orchestrator_skill(user_message: str, current_state: str) -> str:
    result = llm_skill_structured(
        f"""You are an intent classifier for a LinkedIn post generator.

Current workflow state: {current_state}
User message: "{user_message}"

Classify the user's intent into exactly one of these three options:
- new_topic   : user wants to generate a post on a COMPLETELY DIFFERENT subject/topic (e.g. "write about Elon Musk", "now do climate change", "AI")
- feedback    : user wants to MODIFY the current draft — this includes ANY format, length, tone, or style instruction (e.g. "make it shorter", "give me 2 paragraphs", "remove the hashtags", "make it more professional", "add emojis")
- approved    : user is happy with the draft and wants to finalize it (e.g. "yes", "ok", "looks good", "approve", "perfect", "send it")

IMPORTANT RULES:
1. Format/length instructions ("2 paragraphs", "shorter", "longer", "remove tags") are ALWAYS feedback — never new_topic.
2. Only use new_topic when the user clearly names a different subject to write about.
3. When unsure between new_topic and feedback, choose feedback.

Respond with a JSON object with exactly one field:
- "intent": exactly one of "new_topic", "feedback", or "approved" """,
        IntentResult,
    )

    intent = result.intent.strip().lower()

    # Safety: if LLM returns something unexpected, default to feedback
    if intent not in ("new_topic", "feedback", "approved"):
        intent = "feedback"

    return intent
