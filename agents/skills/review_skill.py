from pydantic import BaseModel
from .llm_skill import llm_skill, llm_skill_structured


# Phase 7 and before — manual text parsing (fragile)
# def review_skill(draft, revision_count):
#     response = llm_skill("...DECISION: approved\nCONTENT: ...")
#     decision = "approved"
#     for line in response.strip().split("\n"):
#         if line.startswith("DECISION:"):
#             decision = line.replace("DECISION:", "").strip().lower()
#             break
#     if decision == "approved":
#         content = response.split("CONTENT:", 1)[1].strip() if "CONTENT:" in response else draft
#         return {"decision": "approved", "content": content}
#     else:
#         feedback = response.split("FEEDBACK:", 1)[1].strip() if "FEEDBACK:" in response else ""
#         return {"decision": "needs_work", "content": feedback}


# Phase 8 — Structured Output: Pydantic model guarantees typed response, no parsing needed
class ReviewResult(BaseModel):
    decision: str   # "approved" or "needs_work"
    content: str    # final post if approved, feedback if needs_work


def review_skill(draft: str, revision_count: int = 0) -> dict:
    revision_context = ""
    if revision_count > 0:
        revision_context = f"\nThis is revision #{revision_count}. Be more lenient if it has improved."

    result = llm_skill_structured(
        f"""You are a strict LinkedIn content reviewer.

Draft:
{draft}
{revision_context}

Rules for approval:
- Must be under 250 words
- Must have a strong opening hook
- Must end with a clear call to action

If ALL rules are met: approve and return the polished post.
If ANY rule is violated: reject and return specific actionable feedback.

Respond with a JSON object with exactly these two fields:
- "decision": either "approved" or "needs_work"
- "content": the final polished post (if approved) or specific feedback (if needs_work)""",
        ReviewResult,
    )

    return {"decision": result.decision, "content": result.content}
