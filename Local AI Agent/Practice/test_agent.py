"""
test_agent.py — entry point for running the agent with real user context.
Uses AgentBuilder (Builder Pattern) to configure and compile the agent.
"""
from agentBuilder import AgentBuilder
from modelMiddleware import ModelMiddleware
from weather import Context, locate_user, get_weather
from langchain_core.messages import HumanMessage

USERS = {
    '123': 'Alice',    # Edmonton, Canada
    '456': 'Bob',      # New York City, United States
    '789': 'Charlie',  # London, United Kingdom
}

CONVERSATION = [
    "What is machine learning?",
    "What's the weather like where I am?",           # uses locate_user + get_weather
    "Can you show me code for a linear regression?", 
    "How is that different from deep learning?",
    "And what is the attention mechanism?",
]

def run_for_user(user_id: str) -> None:
    name = USERS.get(user_id, f"user_{user_id}")
    print(f"\n{'='*60}")
    print(f"Session: {name} (user_id={user_id})")
    print('='*60)

    agent = (AgentBuilder()
        .with_middleware(ModelMiddleware.dynamic_model_selection)
        .with_tools(locate_user, get_weather)
        .build())

    config = {
        "configurable": {
            "thread_id": f"session-{user_id}",
            "context": Context(user_id=user_id),
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
