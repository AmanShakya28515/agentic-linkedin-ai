import os
import json
import re
from .llm_skill import llm_skill

# Phase 22 — Dynamic Entity Extraction
# When a document is uploaded, the LLM reads it and extracts the company name,
# product/service names, and people names. These are saved to company_config.json.
# The namespace router reads this file at runtime — no hardcoding needed per client.

# Phase 23 — Railway: use DATA_DIR env var so company_config.json persists on Railway volume
import django.conf
CONFIG_PATH = os.path.join(getattr(django.conf.settings, 'DATA_DIR', os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")), "company_config.json")


def load_config() -> dict:
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"company_names": [], "product_names": [], "people_names": []}


def save_config(config: dict):
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)


def extract_and_save_entities(text: str, doc_type: str):
    """Extract company entities from uploaded text and merge into company_config.json."""
    if doc_type not in ("company_profile", "products", "team_personas"):
        return

    focus = {
        "company_profile": "Focus on the company or brand name.",
        "products":        "Focus on product names, service names, and feature names.",
        "team_personas":   "Focus on people's full names.",
    }

    prompt = f"""You are reading a company document. Extract key entities from it.

Document text:
{text[:2000]}

{focus[doc_type]}

Return ONLY a valid JSON object with exactly these fields:
{{
  "company_names": ["list of company or brand names"],
  "product_names": ["list of product, service, or feature names"],
  "people_names":  ["list of full person names"]
}}

Use empty lists if nothing is found. Return ONLY the JSON, nothing else.

JSON:"""

    try:
        response = llm_skill(prompt)
        match = re.search(r'\{.*?\}', response, re.DOTALL)
        if not match:
            return

        extracted = json.loads(match.group())
        config = load_config()

        for key in ("company_names", "product_names", "people_names"):
            existing_lower = {e.lower() for e in config.get(key, [])}
            for entity in extracted.get(key, []):
                if entity.strip() and entity.lower() not in existing_lower:
                    config.setdefault(key, []).append(entity.strip())
                    existing_lower.add(entity.lower())

        save_config(config)

    except Exception:
        pass
