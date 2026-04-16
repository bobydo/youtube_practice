from dataclasses import dataclass
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage

# ---------------------------------------------------------------------------
# `middleware.py` that demonstrates how to swap an agent's system prompt at runtime
# based on user expertise level (expert / beginner / child). 
# 1. User context dataclass
# ---------------------------------------------------------------------------

@dataclass
class UserRole:
    role: str   # "expert" | "beginner" | "child"
    name: str = "User"

# ---------------------------------------------------------------------------
# 2. Prompt registry — one system prompt per expertise level
# ---------------------------------------------------------------------------

ROLE_PROMPTS = {
    "expert": (
        "You are a precise, technical assistant. "
        "Use domain jargon freely, skip introductory basics, and give depth. "
        "Assume the user has graduate-level knowledge."
    ),
    "beginner": (
        "You are a patient teacher. "
        "Use plain language, everyday analogies, and step-by-step explanations. "
        "Avoid jargon; define any term you must use."
    ),
    "child": (
        "You are a fun, friendly guide talking to a 5-year-old. "
        "Use very simple words, short sentences, and comparisons to toys, "
        "cartoons, or everyday objects the child already knows."
    ),
}

# ---------------------------------------------------------------------------
# 3. dynamic_prompt decorator
# ---------------------------------------------------------------------------

def dynamic_prompt(fn):
    """
    Middleware decorator that intercepts a call to `fn(user_message, user_role)`
    and swaps in the correct system prompt before the request reaches the model.
    """
    def wrapper(user_message: str, user_role: UserRole):
        system_text = ROLE_PROMPTS.get(user_role.role, ROLE_PROMPTS["beginner"])
        messages = [
            SystemMessage(content=system_text),
            HumanMessage(content=user_message),
        ]
        return fn(messages, user_role)
    return wrapper

# ---------------------------------------------------------------------------
# 4. Agent — decorated so the prompt is injected automatically
# ---------------------------------------------------------------------------

@dynamic_prompt
def ask_agent(messages, user_role: UserRole) -> str:
    llm = ChatOllama(model="qwen3:8b")
    response = llm.invoke(messages)
    return response.content

# ---------------------------------------------------------------------------
# 5. Demo
# ---------------------------------------------------------------------------

def main():
    question = "Explain PCA (Principal Component Analysis)."

    roles = [
        UserRole(role="expert",   name="Alice"),
        UserRole(role="beginner", name="Bob"),
        UserRole(role="child",    name="Charlie"),
    ]

    for user in roles:
        print(f"\n{'=' * 60}")
        print(f"User: {user.name}  |  Role: {user.role}")
        print(f"{'=' * 60}")
        answer = ask_agent(question, user)
        print(answer)

if __name__ == "__main__":
    main()
