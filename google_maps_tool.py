from __future__ import annotations

from typing import Dict
import os
import googlemaps
from langchain.tools import Tool


def _get_client(api_key: str | None = None) -> googlemaps.Client:
    api_key = api_key or os.getenv("GOOGLE_MAPS_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_MAPS_API_KEY is not set.")
    return googlemaps.Client(key=api_key)


def get_route(origin: str, destination: str, api_key: str | None = None) -> str:
    """
    Fetch a driving route from Google Maps and return a readable itinerary.
    """
    gmaps = _get_client(api_key)
    directions = gmaps.directions(origin, destination, mode="driving")
    if not directions:
        return "No route found."
    steps = directions[0]["legs"][0]["steps"]
    itinerary_lines = []
    for step in steps:
        # html_instructions may contain HTML; keep simple text fallback
        text = step.get("html_instructions") or step.get("maneuver") or "Proceed"
        distance = step.get("distance", {}).get("text", "")
        itinerary_lines.append(f"{text} for {distance}")
    return "\n".join(itinerary_lines)


def make_route_tool(api_key: str | None = None) -> Tool:
    return Tool(
        name="RouteQuery",
        description="Fetch driving directions between two locations using Google Maps.",
        func=lambda x: get_route(x["origin"], x["destination"], api_key=api_key),
    )

