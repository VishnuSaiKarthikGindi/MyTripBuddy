import requests
import os

class TripAdvisorAPI:
    def __init__(self, api_key, base_url="https://api.content.tripadvisor.com/api/v1"):
        """
        Initialize the API client.

        :param api_key: The API key for accessing TripAdvisor endpoints.
        :param base_url: The base URL for the TripAdvisor API.
        """
        self.api_key = api_key
        self.base_url = base_url

    def _make_request(self, endpoint, params):
        """
        Helper method to make API requests.

        :param endpoint: The API endpoint to call.
        :param params: Query parameters for the request.
        :return: JSON response or None in case of failure.
        """
        headers = {"Accept": "application/json"}
        try:
            response = requests.get(f"{self.base_url}/{endpoint}", headers=headers, params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"API Request failed: {e}")
            return None

    def search_location(self, search_query, category=None, phone=None, address=None, lat_long=None, radius=None, radius_unit=None, language="en"):
        """
        Search for a location by name, category, phone, or coordinates.

        :param search_query: Name of the location to search for.
        :param category: Optional filter by category (e.g., hotels, attractions).
        :param phone: Optional filter by phone number.
        :param address: Optional filter by address.
        :param lat_long: Optional latitude/longitude pair (e.g., "42.3455,-71.10767").
        :param radius: Optional radius for filtering results.
        :param radius_unit: Unit for radius (e.g., km, mi).
        :param language: Language for the results (default is "en").
        :return: JSON response containing location search results.
        """
        endpoint = "location/search"
        params = {
            "key": self.api_key,
            "searchQuery": search_query,
            "category": category,
            "phone": phone,
            "address": address,
            "latLong": lat_long,
            "radius": radius,
            "radiusUnit": radius_unit,
            "language": language
        }
        return self._make_request(endpoint, params)

    def get_location_details(self, location_id, language="en", currency="USD"):
        """
        Get detailed information about a specific location.

        :param location_id: The unique TripAdvisor location ID.
        :param language: Language for the results (default is "en").
        :param currency: Currency for the results (default is "USD").
        :return: JSON response containing location details.
        """
        endpoint = f"location/{location_id}/details"
        params = {
            "key": self.api_key,
            "language": language,
            "currency": currency
        }
        return self._make_request(endpoint, params)

    def nearby_search(self, lat_long, category=None, radius=None, radius_unit="km", language="en"):
        """
        Search for nearby locations based on latitude/longitude.

        :param lat_long: Latitude/Longitude pair (e.g., "42.3455,-71.10767").
        :param category: Optional filter by category (e.g., hotels, attractions).
        :param radius: Optional radius for filtering results.
        :param radius_unit: Unit for radius (default is "km").
        :param language: Language for the results (default is "en").
        :return: JSON response containing nearby search results.
        """
        endpoint = "location/nearby_search"
        params = {
            "key": self.api_key,
            "latLong": lat_long,
            "category": category,
            "radius": radius,
            "radiusUnit": radius_unit,
            "language": language
        }
        return self._make_request(endpoint, params)

# Example usage
if __name__ == "__main__":
    API_KEY = os.getenv("TRIPADVISOR_API_KEY")
    print(API_KEY)
    trip_api = TripAdvisorAPI(API_KEY)

    # Example 1: Search for a location
    search_results = trip_api.search_location("Eiffel Tower", category="attractions")
    print("Search Results:", search_results)

    # Example 2: Get details for a specific location
    if search_results and "data" in search_results:
        location_id = search_results["data"][0]["location_id"]
        details = trip_api.get_location_details(location_id)
        print("Location Details:", details)

    # Example 3: Perform a nearby search
    nearby_results = trip_api.nearby_search("48.8588443,2.2943506", category="hotels", radius=5, radius_unit="km")
    print("Nearby Search Results:", nearby_results)
