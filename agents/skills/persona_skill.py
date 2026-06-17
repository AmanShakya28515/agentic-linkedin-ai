# Phase 18 — Persona-Aware Content
# Defines writing personas for different org roles.
# Same topic, same org data — completely different voice depending on WHO is writing.
#
# CEO post about FlowCore  → company journey, vision, market impact
# Developer post            → how it was built, technical trade-offs
# Sales post                → customer wins, ROI, call to action
# HR post                   → team culture, people behind the product

PERSONAS = {
    "ceo": {
        "role": "CEO & Co-founder",
        "voice": "Strategic visionary who built this company from the ground up. Speaks from lived experience, not theory.",
        "tone_rules": [
            "Write in first person — 'I built', 'I learned', 'We decided'",
            "Reference the company journey, founding decisions, or mission",
            "Focus on market transformation, customer impact, and long-term vision",
            "Use bold, confident statements — you built this, you know it works",
            "Tell the story behind the product, not just the product itself",
        ],
        "avoid": "Technical jargon, detailed pricing, step-by-step instructions",
        "example_opener": "We started this company because I was tired of watching factory managers drown in spreadsheets.",
    },
    "developer": {
        "role": "Senior Developer / Engineer",
        "voice": "Builder who understands the technical depth. Shares how things work, what was hard, what surprised them.",
        "tone_rules": [
            "Write from a builder's perspective — 'here is how we built it', 'here is what we learned'",
            "Use specific technical details — mention machine types, protocols, architecture decisions",
            "Be honest about challenges — 'the hardest part was...'",
            "Explain complexity in simple terms — respect the reader's intelligence",
            "Reference real engineering trade-offs and decisions",
        ],
        "avoid": "Sales language, vague claims, motivational fluff",
        "example_opener": "Connecting a 30-year-old Siemens machine to a cloud dashboard should not take 6 months. We made it take 4 hours.",
    },
    "sales": {
        "role": "Sales Manager / Account Executive",
        "voice": "Results-driven. Speaks in customer wins, ROI, and specific outcomes. Every sentence pushes toward a decision.",
        "tone_rules": [
            "Lead with a specific customer result — a real number, a real outcome",
            "Use problem-solution-result structure in every post",
            "Make the value proposition clear in the first 2 sentences",
            "Use specific numbers always — never say 'significant improvement'",
            "End with a direct call to action — 'book a demo', 'let us talk', 'reply YES'",
        ],
        "avoid": "Technical details, founding stories, philosophical statements",
        "example_opener": "One of our customers replaced 3 data entry staff in 6 weeks. Here is how.",
    },
    "hr": {
        "role": "HR Manager / People Lead",
        "voice": "People-first. Writes about culture, team dynamics, and the human side of work. Warm but direct.",
        "tone_rules": [
            "Focus on the people behind the product — team culture, values, how we work",
            "Write about hiring, growth, and what it means to work here",
            "Use 'our team', 'our people', 'we believe'",
            "Reference company values authentically — no corporate speak",
            "Make readers understand why the team is special or want to join",
        ],
        "avoid": "Product features, technical details, pricing, sales language",
        "example_opener": "We are 87 people across 3 cities. The thing that keeps us together is the problem we are solving.",
    },
}


def get_persona_prompt(persona_key: str) -> str:
    """Return persona instructions to inject into the writing prompt.
    Returns empty string for 'default' or unknown keys — no persona injection."""
    persona = PERSONAS.get(persona_key)
    if not persona:
        return ""

    rules = "\n".join(f"  - {r}" for r in persona["tone_rules"])

    return f"""
PERSONA — Write as this specific person (overrides generic LinkedIn creator style):
Role: {persona["role"]}
Voice: {persona["voice"]}
Tone rules:
{rules}
Avoid: {persona["avoid"]}
Example opener style: "{persona["example_opener"]}"
"""
