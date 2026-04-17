"""
test_agent.py — entry point for running the agent with real user context.

Imports dynamicModel and weather as libraries:
  - dynamicModel: agent graph, hooks, model factory
  - weather:      tools (locate_user, get_weather) + user Context
"""
from dynamicModel import (
    create_agent,
    dynamic_model_selection,
    HookRegistry,
    on_before_agent,
    on_before_model,
    on_after_model,
    on_after_agent,
    on_error,
)
from weather import Context, locate_user, get_weather
from langchain_core.messages import HumanMessage

# ---------------------------------------------------------------------------
# Test users — maps user_id to a display name for readable output
# ---------------------------------------------------------------------------

USERS = {
    '123': 'Alice',    # Edmonton, Canada
    '456': 'Bob',      # New York City, United States
    '789': 'Charlie',  # London, United Kingdom
}

# ---------------------------------------------------------------------------
# Test conversations — mix of general, weather, and code questions
# ---------------------------------------------------------------------------

CONVERSATION = [
    "What is machine learning?",
    "What's the weather like where I am?",          # uses locate_user + get_weather
    "Can you show me code for a linear regression?", # routes to code_agent
    "How is that different from deep learning?",
    "And what is the attention mechanism?",
]

def run_for_user(user_id: str) -> None:
    name = USERS.get(user_id, f"user_{user_id}")
    print(f"\n{'='*60}")
    print(f"Session: {name} (user_id={user_id})")
    print('='*60)

    # even hooks; on is append fn, emit is trigger event with payload; agent will call emit at the right times
    hooks = HookRegistry()
    hooks.on("before_agent", on_before_agent)
    hooks.on("before_model", on_before_model)
    hooks.on("after_model",  on_after_model)
    hooks.on("after_agent",  on_after_agent)
    hooks.on("on_error",     on_error)

    # Pass tools here — dynamicModel stays tool-agnostic
    agent = create_agent(
        middleware=[dynamic_model_selection],
        hooks=hooks,
        tools=[locate_user, get_weather],
    )

    config = {
        "configurable": {
            "thread_id": f"session-{user_id}",
            "context": Context(user_id=user_id),   # gives locate_user access to user_id
        }
    }

    for turn, user_input in enumerate(CONVERSATION, start=1):
        print(f"\n--- Turn {turn} ---")
        print(f"{name}: {user_input}")

        result = agent.invoke(
            {"messages": [HumanMessage(content=user_input)]},
            config=config,
        )

        reply = result["messages"][-1].content
        print(f"Agent: {reply[:160]}{'...' if len(reply) > 160 else ''}")


if __name__ == "__main__":
    run_for_user("123")   # change to "456" or "789" to test other users
