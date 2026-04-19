import csv
import os
from dataclasses import dataclass
from dotenv import load_dotenv
from pydantic import BaseModel

from langchain_core.runnables import RunnableConfig
from langchain.tools import tool
from loggerSetup import get_logger

load_dotenv()

logger = get_logger(__name__)

@dataclass
class Context:
    user_id: str

class ResponseFormat(BaseModel):
    summary: str
    temperature: float
    temperature_fahrenheit: float
    humidity: float

USER_LOCATIONS: dict[str, int] = {
    '123': 5946768,   # Edmonton, Canada
    '456': 5128581,   # New York City, United States
    '789': 2643743,   # London, United Kingdom
}

CITIES_CSV = os.path.join(os.path.dirname(__file__), '..', 'data', 'world-cities.csv')

# ---------------------------------------------------------------------------
# 1. LocationService — Strategy pattern
#    Owns user → city lookup. CSV is one strategy, swappable via injection.
# ---------------------------------------------------------------------------

class LocationService:
    def __init__(self, user_locations: dict[str, int], cities_csv: str):
        self._user_locations = user_locations
        self._cities_csv = cities_csv

    def lookup(self, user_id: str) -> str:
        geonameid = self._user_locations.get(user_id)
        if geonameid is None:
            logger.warning("location_lookup  user_id=%s  result=Unknown", user_id)
            return 'Unknown'
        with open(self._cities_csv, newline='', encoding='utf-8') as f:
            for row in csv.DictReader(f):
                if row['geonameid'] == str(geonameid):
                    city_country = f"{row['name']}, {row['country']}"
                    logger.info("location_lookup  user_id=%s  city=%s", user_id, city_country)
                    return city_country
        return 'Unknown'

# ---------------------------------------------------------------------------
# 2. WeatherApi — Adapter pattern
#    Wraps wttr.in. Swap weather provider by replacing this class.
# ---------------------------------------------------------------------------

class WeatherApi:
    def fetch(self, city: str, country: str) -> dict:
        import requests
        logger.info("weather_fetch  city=%s  country=%s", city, country)
        response = requests.get(f"https://wttr.in/{city},{country}?format=j1")
        return response.json()

# ---------------------------------------------------------------------------
# 3. WeatherService — Facade pattern
#    Hides the locate → fetch chain behind one method.
# ---------------------------------------------------------------------------

class WeatherService:
    def __init__(self, location: LocationService, weather: WeatherApi):
        self._location = location
        self._weather = weather

    def get_for_user(self, user_id: str) -> dict:
        city_country = self._location.lookup(user_id)
        if city_country == 'Unknown':
            return {'error': 'User location not found'}
        city, country = city_country.split(', ', 1)
        return self._weather.fetch(city, country)

# ---------------------------------------------------------------------------
# 4. Module-level instances
# ---------------------------------------------------------------------------

_location_service = LocationService(USER_LOCATIONS, CITIES_CSV)
_weather_api = WeatherApi()
_weather_service = WeatherService(_location_service, _weather_api)

# ---------------------------------------------------------------------------
# 5. @tool wrappers — Adapter pattern (thin, no logic)
#    Keep both tools separate so LLM can skip locate_user when user
#    provides city directly ("weather in Tokyo").
# ---------------------------------------------------------------------------

@tool('locate_user', description="Automatically returns the current user's city — no input required. Call this first whenever you need to know where the user is located.")
def locate_user(config: RunnableConfig) -> str:
    """Return the city for the current user."""
    context: Context = config.get('configurable', {}).get('context', Context(user_id=''))
    return _location_service.lookup(context.user_id)

@tool(
    'get_weather',
    description=(
        "Use this tool ONLY when the user asks about weather, temperature, "
        "forecast, humidity, or what it is like outside in a specific location."
    ),
    return_direct=False,
)
def get_weather(city: str, country: str) -> dict:
    """Get current weather for a city. Requires BOTH city and country."""
    return _weather_api.fetch(city, country)
