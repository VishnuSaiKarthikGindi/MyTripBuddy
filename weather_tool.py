from __future__ import annotations

from typing import Optional
from langchain_community.utilities import OpenWeatherMapAPIWrapper
from langchain.tools import Tool


def make_weather_tool(api_key: Optional[str] = None) -> Tool:
    """
    Returns a LangChain Tool for querying OpenWeatherMap by location name.

    If api_key is None, the wrapper will read OPENWEATHERMAP_API_KEY from the environment.
    """
    wrapper = OpenWeatherMapAPIWrapper(
        openweathermap_api_key=api_key,
        top_k_results=1,
        doc_content_chars_max=300,
    )
    return Tool(
        name="WeatherQuery",
        description="Fetch current weather for a city/location using OpenWeatherMap.",
        func=wrapper.run,
    )

