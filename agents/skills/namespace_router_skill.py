import json
import re
from .llm_skill import llm_skill

# Phase 17 — Namespace Router
# LLM reads the user's post request and decides which org knowledge collections
# are relevant to retrieve from. This prevents mixing irrelevant data into the
# writing context (e.g., pulling team bios when writing about a product feature).

NAMESPACES = {
    "company_profile":  "Company info, mission, values, history, founding story",
    "products":         "Product/service descriptions, features, pricing, use cases",
    "brand_guidelines": "Tone of voice, writing style, brand rules, visual identity",
    "audience":         "Target audience, ICP, customer segments, customer pain points",
    "campaigns":        "Past campaigns, content calendar, events, product launches, performance data",
}


def route_namespaces(topic: str) -> list:
    """Given a post topic/request, return the relevant namespace keys to search."""
    ns_list = "\n".join(f"- {k}: {v}" for k, v in NAMESPACES.items())

    # Phase 22 — load company entities dynamically from company_config.json
    # No hardcoding — entities are extracted automatically when documents are uploaded
    from .entity_extractor_skill import load_config
    config = load_config()
    KNOWN_PRODUCTS = config.get("product_names", [])
    KNOWN_COMPANY  = config.get("company_names", [])
    KNOWN_PEOPLE   = config.get("people_names", [])

    products_str = ", ".join(KNOWN_PRODUCTS) if KNOWN_PRODUCTS else "none configured"
    company_str  = ", ".join(KNOWN_COMPANY)  if KNOWN_COMPANY  else "none configured"
    people_str   = ", ".join(KNOWN_PEOPLE)   if KNOWN_PEOPLE   else "none configured"

    prompt = f"""You are a knowledge routing agent for an organizational AI system.

Your job: given a LinkedIn post request, decide which organizational knowledge namespaces are relevant.

Available namespaces:
{ns_list}

Post request: {topic}

KNOWN company entities (the ONLY things that should trigger substantive namespaces):
- Products/Services: {products_str}
- Company names:     {company_str}
- People:            {people_str}

Rules:
- Always include "brand_guidelines" — applies to every post
- Only include "products" if the request mentions one of the known products or services above
- Only include "company_profile" if the request is about the company itself (its mission, story, values, or overview)
- Only include "team_personas" if the request mentions a known person or asks about the team
- Only include "audience" if the request mentions customers, target market, or ICP
- Only include "campaigns" if the request mentions campaign performance, data, results, or metrics
- If the topic is about a general person (celebrity, athlete, public figure), a general concept, or current events — return ONLY ["brand_guidelines"]
- Return at most 3 namespaces
- Return ONLY a valid JSON array of namespace keys, nothing else

Examples:
- "Write about our lead generation feature"  → ["products", "brand_guidelines"]
- "Tell our company story"                   → ["company_profile", "brand_guidelines"]
- "Write about our campaign results"         → ["campaigns", "brand_guidelines"]
- "Write about Messi's discipline"           → ["brand_guidelines"]
- "Write about Conor McGregor"               → ["brand_guidelines"]
- "Write about our target customers"         → ["audience", "brand_guidelines"]

JSON array:"""

    try:
        response = llm_skill(prompt)
        match = re.search(r'\[.*?\]', response, re.DOTALL)
        if match:
            namespaces = json.loads(match.group())
            valid = [n for n in namespaces if n in NAMESPACES]
            if valid:
                return valid
    except Exception:
        pass

    # Fallback: rule-based routing using dynamically loaded entities
    lower = topic.lower()
    result = ["brand_guidelines"]

    known_company_lower  = [c.lower() for c in KNOWN_COMPANY]
    known_products_lower = [p.lower() for p in KNOWN_PRODUCTS]
    known_people_lower   = [p.lower() for p in KNOWN_PEOPLE]

    if any(c in lower for c in known_company_lower) and any(p in lower for p in known_products_lower):
        result.append("products")
    elif any(c in lower for c in known_company_lower):
        result.append("company_profile")
    elif any(w in lower for w in ["customer", "audience", "client", "our users", "segment", "icp"]):
        result.append("audience")
    elif any(w in lower for w in ["campaign", "event", "calendar", "launch", "results", "metrics", "performance"]):
        result.append("campaigns")

    return result
