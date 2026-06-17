from pydantic import BaseModel
from typing import List
from .llm_skill import llm_skill_structured

# Phase 19 — Brand Voice Enforcer
# Checks a generated draft against the org's uploaded brand guidelines.
# Unlike the Phase 8 review agent (which checks writing quality),
# this enforcer checks COMPLIANCE — does this post follow THIS company's specific rules?
#
# Key difference: rules come from the uploaded brand guidelines document,
# not from hardcoded logic. Change the doc → change the rules. No code update needed.


class BrandCheckResult(BaseModel):
    passed: bool              # True if no violations found
    violations: List[str]     # specific rules broken (empty if passed)
    rewrite_instruction: str  # exact instruction for the writer to fix it (empty if passed)


def check_brand_compliance(draft: str, brand_guidelines: str) -> dict:
    """Check draft against uploaded brand guidelines.
    Returns dict with passed, violations, rewrite_instruction.
    Auto-passes if no brand guidelines are uploaded yet."""

    if not brand_guidelines or not brand_guidelines.strip():
        return {"passed": True, "violations": [], "rewrite_instruction": ""}

    result = llm_skill_structured(
        f"""You are a strict brand compliance checker for a company's LinkedIn content.

BRAND GUIDELINES (these are the rules — follow them exactly):
{brand_guidelines}

DRAFT TO CHECK:
{draft}

Your job: check if this draft violates any rules in the brand guidelines.

Be very strict. Check for:
- Banned words AND their variations/derivatives. If "revolutionary" is banned, also flag "revolutionize", "revolutionizing", "revolutionized". If "game-changer" is banned, also flag "game changer", "game-changing", "game changing".
- Tone violations (wrong voice, wrong person, corporate speak)
- Format violations (too many hashtags, run-on sentences, wrong structure)
- Content violations (wrong audience framing, missing required elements)

Do NOT give the benefit of the doubt. If a word is a variation of a banned word — flag it.

Respond with a JSON object with exactly these three fields:
- "passed": true only if ZERO violations found, false if ANY violation found
- "violations": a JSON array listing each specific violation found (e.g. ["Used banned word variation: revolutionizing (banned: revolutionary)", "Used banned phrase: game-changer", "More than 5 hashtags"])
- "rewrite_instruction": if failed, one clear instruction telling the writer exactly which words to remove and what to replace them with; empty string if passed""",
        BrandCheckResult,
    )

    return {
        "passed": result.passed,
        "violations": result.violations,
        "rewrite_instruction": result.rewrite_instruction,
    }
