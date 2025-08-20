from typing import Dict, Any, Type
from pydantic import BaseModel, Field
from langchain.tools import BaseTool
import ast
import os
import re
from trip_api import TripAdvisorAPI
import json

# Define Pydantic schema for tool arguments
class SearchAccommodationAndAttractionsArgs(BaseModel):
    query: str = Field(..., description="Natural language query for accommodations or attractions.")

# Define the tool class
class SearchAccommodationAndAttractions(BaseTool):
    name: str = "SearchAccommodationAndAttractions"
    description: str = (
                "Search for hotels, attractions, and restaurants using the TripAdvisor API. "
                "Input should be a dictionary with keys: 'location' (required), 'type' (required: 'hotels', 'attractions', or 'restaurants'), "
                "and optionally 'radius', 'radius_unit', and 'language'. Returns relevant information about the specified type near the given location."
    )

    args_schema: Type[BaseModel] = SearchAccommodationAndAttractionsArgs
    api_client: TripAdvisorAPI

    def _run(self, query: str) -> Dict[str, Any]:
      def parse_user_query(query: str) -> Dict[str, Any]:
          parsed_query = {"language": "en"}

          try:
              # Attempt to parse the query as a dictionary-like string
              query_dict = ast.literal_eval(query)
              if isinstance(query_dict, dict):
                  parsed_query["searchQuery"] = query_dict.get("location", "")
                  parsed_query["category"] = query_dict.get("type", "")
                  parsed_query["radius"] = query_dict.get("radius", None)
                  parsed_query["radiusUnit"] = query_dict.get("radiusUnit", None)
                  return parsed_query
          except (ValueError, SyntaxError):
              # If the query is not a dictionary-like string, proceed with natural language parsing
              pass

          # Define patterns for different query types
          patterns = {
              "top_attractions": r"top attractions in (?P<location>.+)",
              "must_visit": r"must-visit places in (?P<location>.+)",
              "best_things_to_do": r"best things to do in (?P<location>.+) for (?P<type>families|couples|solo travelers)",
              "budget_friendly": r"free or budget-friendly attractions in (?P<location>.+)",
              "hidden_gems": r"hidden gems in (?P<location>.+)",
              "cultural_landmarks": r"cultural or historical landmarks in (?P<location>.+)",
              "outdoor_activities": r"outdoor activities or natural attractions in (?P<location>.+)",
              "best_time_to_visit": r"best time to visit (?P<attraction>.+)",
              "crowded_times": r"is (?P<attraction>.+) crowded during weekends/holidays",
              "best_season": r"best season to visit (?P<location>.+)"
          }

          # Extract information using regex patterns
          for key, pattern in patterns.items():
              match = re.search(pattern, query, re.IGNORECASE)
              if match:
                  parsed_query.update(match.groupdict())
                  if key in ["best_things_to_do", "cultural_landmarks", "outdoor_activities", "top_attractions", "must_visit", "hidden_gems", "best_time_to_visit", "crowded_times", "best_season"]:
                      parsed_query["category"] = "attractions"
                  elif key == "budget_friendly":
                      parsed_query["category"] = "attractions"
                  if "location" in parsed_query and not parsed_query.get("searchQuery"):
                      parsed_query["searchQuery"] = parsed_query["location"]
                  if "attraction" in parsed_query and not parsed_query.get("searchQuery"):
                      parsed_query["searchQuery"] = parsed_query["attraction"]
                  return parsed_query

          tokens = query.lower().split()
          categories = {"hotels", "attractions", "restaurants"}

          # Extract category from the query
          for category in categories:
              if category in tokens:
                  parsed_query["category"] = category
                  break

          # Extract location or POI
          if "near" in tokens:
              idx = tokens.index("near") + 1
              if idx < len(tokens):
                  parsed_query["searchQuery"] = " ".join(tokens[idx:])

          # Extract radius if mentioned
          if "within" in tokens:
              idx = tokens.index("within") + 1
              try:
                  parsed_query["radius"] = int(tokens[idx])
                  if idx + 1 < len(tokens) and tokens[idx + 1] in {"km", "mi"}:
                      parsed_query["radiusUnit"] = tokens[idx + 1]
              except (ValueError, IndexError):
                  pass

          if not parsed_query.get("searchQuery"):
              parsed_query["searchQuery"] = query
          return parsed_query

      query_params = parse_user_query(query)
      if not self.api_client:
          return {"error": "TripAdvisor API client not configured."}
      if "latLong" in query_params:
          results = self.api_client.nearby_search(
              lat_long=query_params.get("latLong"),
              category=query_params.get("category"),
              radius=query_params.get("radius"),
              radius_unit=query_params.get("radiusUnit"),
              language=query_params.get("language"),
          )
      else:
          results = self.api_client.search_location(
                search_query=query_params.get("searchQuery"),
                category=query_params.get("category"),
                radius=query_params.get("radius"),
                radius_unit=query_params.get("radiusUnit"),
                language=query_params.get("language"),
          )
      return results or {"error": "No results or API error."}

    def _arun(self, query: str) -> Dict[str, Any]:
        raise NotImplementedError("This tool does not support async")

# Initialize the tool with the API client
tripadvisor_api = TripAdvisorAPI(api_key=os.getenv("TRIPADVISOR_API_KEY"))
search_tool = SearchAccommodationAndAttractions(api_client=tripadvisor_api)