from dataclasses import dataclass
from dotenv import load_dotenv
from pydantic import BaseModel

from langchain_ollama import ChatOllama
from langchain_core.runnables import RunnableConfig
from langgraph.prebuilt import create_react_agent
from langchain.tools import tool
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()

@dataclass
class Context:
    user_id: str

class ResponseFormat(BaseModel):
    summary: str
    temperature: float
    temperature_fahrenheit: float
    humidity: float

llm = ChatOllama(
    model="qwen3:8b",
    temperature=0.3,
    base_url="http://localhost:11434"
)

@tool('locate_user', description="Look up a user's city based on their user_id in context")
def locate_user(config: RunnableConfig) -> str:
    """Return the city for the current user."""
    context: Context = config.get('configurable', {}).get('context', Context(user_id=''))
    user_id = context.user_id
    match user_id:
        case '123':
            return 'Edmonton, Canada'
        case '456':
            return 'New York, USA'
        case '789':
            return 'London, UK'
        case _:
            return 'Unknown'

@tool('get_weather', description='Return weather information for a given city', return_direct=False)
def get_weather(city: str, country: str):
    """Get current weather for a city. Requires BOTH city and country."""
    import requests
    response = requests.get(f"https://wttr.in/{city},{country}?format=j1")
    return response.json()

system_prompt = """You are a helpful AI assistant that can use tools to answer user questions.

GENERAL RULES:

Be concise and accurate.
Use tools when needed; do not guess when real data is required.
If you already know the answer with high confidence, respond directly.

TOOL USAGE:

Only call a tool when it clearly helps answer the question.
Always follow the tool's required input format exactly.
Do not fabricate tool inputs.

AMBIGUITY HANDLING:

If the user request is ambiguous (e.g., a city name shared by multiple locations),
DO NOT guess.
Ask a clear clarification question before calling any tool."""

checkpoint = MemorySaver()

agent = create_react_agent(
    model=llm,
    tools=[locate_user, get_weather],
    prompt=system_prompt,
    checkpointer=checkpoint,
)

structured_llm = llm.with_structured_output(ResponseFormat)

response = agent.invoke(
    {"messages": [{"role": "user", "content": "What's the weather like where I am?"}]},
    config={"configurable": {"thread_id": "1", "context": Context(user_id="123")}}
)

final_text = response['messages'][-1].content
structured = structured_llm.invoke(
    f"Extract weather data from this text into the required fields:\n\n{final_text}"
)

print(final_text)

print("\n===== STRUCTURED RESPONSE =====")
print(structured.summary)
print(structured.temperature)
print(structured.temperature_fahrenheit)
print(structured.humidity)