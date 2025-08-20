from __future__ import annotations

from typing import Optional, List, Dict, Any, Tuple
import os
import re
import json

from langchain.tools import Tool

from SearchPOITool import search_tool as tripadvisor_search_tool
from weather_tool import make_weather_tool
from google_maps_tool import make_route_tool, get_route
from vector_search import VectorSearch, make_vector_search_tool

try:
    from amadeus_tool import get_amadeus_tools
except Exception:  # Amadeus is optional
    get_amadeus_tools = None  # type: ignore


class Orchestrator:
    """
    Routes user queries to the appropriate tool and returns the response.
    Tools included:
      - TripAdvisor search (accommodations/attractions/restaurants)
      - Weather (OpenWeatherMap)
      - Google Maps directions
      - Vector search (Pinecone + embeddings)
      - Amadeus (if credentials available)
    """

    def __init__(
        self,
        include_amadeus: bool = True,
        openai_api_key: Optional[str] = None,
        pinecone_api_key: Optional[str] = None,
        vector_index_name: str = "mytripbuddy",
    ) -> None:
        self.tools: Dict[str, Any] = {}

        # TripAdvisor search tool (already initialized by module)
        self.tools["tripadvisor"] = tripadvisor_search_tool

        # Weather tool
        self.tools["weather"] = make_weather_tool()

        # Google Maps route tool
        self.tools["route_tool"] = make_route_tool()

        # Vector search (optional but recommended)
        self.vector_search: Optional[VectorSearch] = None
        try:
            self.vector_search = VectorSearch(
                index_name=vector_index_name,
                openai_api_key=openai_api_key or os.getenv("OPENAI_API_KEY"),
                pinecone_api_key=pinecone_api_key or os.getenv("PINECONE_API_KEY"),
            )
            self.tools["vector_tool"] = make_vector_search_tool(self.vector_search)
        except Exception:
            # If not configured, skip vector search
            self.tools["vector_tool"] = None

        # Amadeus tools (optional)
        self.amadeus_tools: List[Tool] = []
        if include_amadeus and get_amadeus_tools is not None:
            try:
                self.amadeus_tools = get_amadeus_tools()
            except Exception:
                self.amadeus_tools = []

    # ----------------------- Routing ----------------------- #
    def _parse_route(self, query: str) -> Optional[Tuple[str, str]]:
        text = query.strip()
        # from A to B
        m = re.search(r"from\s+(?P<origin>.+?)\s+to\s+(?P<dest>.+)$", text, re.I)
        if m:
            return m.group("origin"), m.group("dest")
        # between A and B
        m = re.search(r"between\s+(?P<origin>.+?)\s+and\s+(?P<dest>.+)$", text, re.I)
        if m:
            return m.group("origin"), m.group("dest")
        # A to B (short form)
        m = re.search(r"^(?P<origin>.+?)\s+to\s+(?P<dest>.+)$", text, re.I)
        if m:
            return m.group("origin"), m.group("dest")
        return None

    def _should_use_weather(self, query: str) -> bool:
        q = query.lower()
        return any(k in q for k in ["weather", "temperature", "forecast", "rain", "snow"])

    def _should_use_route(self, query: str) -> bool:
        q = query.lower()
        return any(k in q for k in ["route", "directions", "drive", "driving"]) or bool(self._parse_route(query))

    def _should_use_tripadvisor(self, query: str) -> bool:
        q = query.lower()
        return any(k in q for k in [
            "hotel", "hotels", "attraction", "attractions", "restaurant", "restaurants",
            "things to do", "places to visit", "nearby", "top attractions", "must-visit",
        ])

    def _should_use_amadeus(self, query: str) -> bool:
        q = query.lower()
        return any(k in q for k in ["flight", "flights", "fare", "airline", "ticket", "itinerary"]) and bool(self.amadeus_tools)

    # ----------------------- Execution ----------------------- #
    def route_and_execute(self, query: str) -> str:
        # Flights / Amadeus
        if self._should_use_amadeus(query):
            for tool in self.amadeus_tools:
                try:
                    return str(tool.run(query))
                except Exception:
                    continue
            return "Could not process the flight-related request with Amadeus tools."

        # Weather
        if self._should_use_weather(query):
            weather_tool = self.tools.get("weather")
            try:
                return str(weather_tool.run(query))  # type: ignore
            except Exception as e:
                return f"Weather tool error: {e}"

        # Route
        if self._should_use_route(query):
            od = self._parse_route(query)
            if od:
                origin, dest = od
                try:
                    return get_route(origin, dest)
                except Exception as e:
                    return f"Route tool error: {e}"
            # If no parse, ask user format; still try tool with a helpful message
            return "Please provide route as: 'from ORIGIN to DESTINATION'"

        # TripAdvisor places
        if self._should_use_tripadvisor(query):
            ta_tool = self.tools.get("tripadvisor")
            try:
                result = ta_tool.run(query)  # type: ignore
                # Normalize to string
                if isinstance(result, (dict, list)):
                    return json.dumps(result, indent=2)
                return str(result)
            except Exception as e:
                return f"TripAdvisor tool error: {e}"

        # Vector search fallback
        vector_tool = self.tools.get("vector_tool")
        if vector_tool is not None:
            try:
                return str(vector_tool.run(query))  # type: ignore
            except Exception as e:
                return f"Vector search error: {e}"

        # Last resort
        return "No suitable tool available to answer this query. Please refine your request."


def build_default_orchestrator() -> Orchestrator:
    return Orchestrator()


if __name__ == "__main__":
    orch = build_default_orchestrator()
    print(orch.route_and_execute("top attractions in Paris within 5 km"))
