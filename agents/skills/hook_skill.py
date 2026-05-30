from .llm_skill import llm_skill


def hook_skill(research: str) -> str:
    """
    Takes research text and generates 3 strong
    LinkedIn hooks to grab attention.
    """
    return llm_skill(
        f"Create 3 strong LinkedIn hooks based on this research:\n{research}"
    )
