import os
import json
from dotenv import load_dotenv
from openai import OpenAI
from langsmith import traceable   # Phase 20 — observability

load_dotenv()

# Phase 3/4 — OpenRouter (nvidia/nemotron-3-nano-30b-a3b:free, 50 req/day free limit)
# client = OpenAI(
#     base_url="https://openrouter.ai/api/v1",
#     api_key=os.getenv("OPEN_ROUTER_KEY"),
# )
# MODEL = "nvidia/nemotron-3-nano-30b-a3b:free"

# Phase 5 — Groq (llama-3.3-70b-versatile, 14,400 req/day free, faster)
client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.getenv("GROQ_API_KEY"),
)

MODEL = "llama-3.3-70b-versatile"


# Phase 20 — @traceable sends this call to LangSmith with prompt, response, and token count
@traceable(name="llm_skill", run_type="llm")
def llm_skill(prompt: str) -> str:
    """
    Base skill — sends a prompt to the LLM and returns the response.
    All other skills call this. To swap the model or provider,
    change this one file — nothing else needs to change.
    """
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content


# Phase 8 — Structured Output: LLM returns JSON, validated into a Pydantic model
# Before: skills manually parsed text ("DECISION: approved\n...") — fragile
# After:  LLM guaranteed to return JSON → Pydantic validates → typed object, no parsing
# Phase 20 — @traceable tracks structured calls separately so you can filter by type
@traceable(name="llm_skill_structured", run_type="llm")
def llm_skill_structured(prompt: str, schema_class):
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
    )
    data = json.loads(response.choices[0].message.content)
    return schema_class(**data)


# Phase 6 — Streaming: yields tokens one by one instead of waiting for full response
# Phase 20 — streaming traces appear in LangSmith as a single run when complete
@traceable(name="llm_skill_stream", run_type="llm")
def llm_skill_stream(prompt: str):
    stream = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        stream=True,
    )
    for chunk in stream:
        token = chunk.choices[0].delta.content
        if token:
            yield token
