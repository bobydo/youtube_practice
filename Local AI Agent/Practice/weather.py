import csv
import os
from dataclasses import dataclass
from dotenv import load_dotenv
from pydantic import BaseModel
from langsmith import traceable

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

# Maps user_id → geonameid from data/world-cities.csv
# To add a new user: find their city's geonameid in the CSV and add it here
USER_LOCATIONS: dict[str, int] = {
    '123': 5946768,   # Edmonton, Canada
    '456': 5128581,   # New York City, United States
    '789': 2643743,   # London, United Kingdom
}

CITIES_CSV = os.path.join(os.path.dirname(__file__), '..', 'data', 'world-cities.csv')

@traceable(name="lookup_city")
def _lookup_city(geonameid: int) -> str:
    """Return 'City, Country' by scanning world-cities.csv for a matching geonameid."""
    with open(CITIES_CSV, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            if row['geonameid'] == str(geonameid):
                return f"{row['name']}, {row['country']}"
    return 'Unknown'

@tool('locate_user', description="Automatically returns the current user's city — no input required. Call this first whenever you need to know where the user is located.")
def locate_user(config: RunnableConfig) -> str:
    """Return the city for the current user."""
    context: Context = config.get('configurable', {}).get('context', Context(user_id=''))
    geonameid = USER_LOCATIONS.get(context.user_id)
    if geonameid is None:
        return 'Unknown'
    return _lookup_city(geonameid)

@tool(
    'get_weather',
    description=(
        "Use this tool ONLY when the user asks about weather, temperature, "
        "forecast, humidity, or what it is like outside in a specific location."
    ),
    return_direct=False,
)
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
    response_format=ResponseFormat,
)

# @traceable makes this entire run appear as a root span in LangSmith.
# Returning the dict logs it as the span output — visible in the LangSmith UI
# instead of printed to the terminal.
@traceable(name="weather_run")
def run(user_id: str) -> dict:
    response = agent.invoke(
        {"messages": [{"role": "user", "content": "What's the weather like where I am?"}]},
        config={"configurable": {"thread_id": "1", "context": Context(user_id=user_id)}}
    )
    structured: ResponseFormat = response['structured_response']
    return {
        "summary": structured.summary,
        "temperature_c": structured.temperature,
        "temperature_f": structured.temperature_fahrenheit,
        "humidity": structured.humidity,
    }

if __name__ == "__main__":
    result = run(user_id="123")
    print(result)