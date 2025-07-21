from dotenv import load_dotenv
from amadeus import Client, ResponseError
from geopy.geocoders import Nominatim
import os, requests

load_dotenv()

#AVIATIONSTACK_API_KEY = os.getenv("AVIATIONSTACK_API_KEY")
AMADEUS_CLIENT_ID = os.getenv("AMADEUS_CLIENT_ID")
AMADEUS_CLIENT_SECRET = os.getenv("AMADEUS_CLIENT_SECRET")
#OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")  Not useful for free tier
amadeus = Client(
        client_id=AMADEUS_CLIENT_ID,
        client_secret=AMADEUS_CLIENT_SECRET
                )

# --------- FLIGHT API ----------
# Uncomment the following lines if you want to use AviationStack API (but unreliable for future flights and limited calls)
# def get_flights_data(departure, arrival, date=None):
#     url = "http://api.aviationstack.com/v1/flights"
#     params = {
#         "access_key": AVIATIONSTACK_API_KEY,
#         "dep_iata": departure,
#         "arr_iata": arrival,
#         "flight_status": "scheduled",
#         "flight_date": date  # if needed
#     }
#     try:
#         res = requests.get(url, params=params)
#         res.raise_for_status()
#         return res.json().get("data", [])[:3]
#     except Exception as e:
#         return [{"error": str(e)}]
def get_flights_data(departure, arrival, date=None):
    try:
        response = amadeus.shopping.flight_offers_search.get(
            originLocationCode=departure.upper(),
            destinationLocationCode=arrival.upper(),
            departureDate=date,
            adults=1,
            max=3,
            currencyCode="INR"
        )

        flights = response.data
        top_flights = []

        for flight in flights:
            itinerary = flight['itineraries'][0]
            segments = itinerary['segments']
            duration = itinerary['duration']
            price = flight['price']['total']

            top_flights.append({
                "price": f"{price} INR",
                "duration": duration,
                "segments": [
                    {
                        "from": seg['departure']['iataCode'],
                        "to": seg['arrival']['iataCode'],
                        "airline": seg['carrierCode'],
                        "flightNumber": seg['number'],
                        "departureTime": seg['departure']['at'],
                        "arrivalTime": seg['arrival']['at']
                    } for seg in segments
                ]
            })

        return top_flights

    except ResponseError as e:
        return [{"error": str(e)}]

# --------- HOTEL API (Amadeus) ----------
def get_hotels_data(city_code: str, checkin_date: str, checkout_date: str, adults: int = 1):
    """
    Implements the two-phase hotel search to fetch offers for a given city.

    This function first uses the Hotel List API to find hotel IDs for a city,
    and then uses the Hotel Search API to get real-time offers for those hotels.

    Args:
        amadeus: An initialized Amadeus API client instance.
        city_code: The IATA code for the city (e.g., 'PAR').
        checkin_date: The check-in date in 'YYYY-MM-DD' format.
        checkout_date: The check-out date in 'YYYY-MM-DD' format.
        adults: The number of adults per room.

    Returns:
        A list of hotel offer dictionaries, or an error dictionary if an issue occurs.
    """
    # --- Phase 1: Get Hotel IDs from City Code ---
    # This call translates the city code into a list of specific hotel properties.
    try:
        hotel_list_response = amadeus.reference_data.locations.hotels.by_city.get(
            cityCode=city_code
        )
        hotels = hotel_list_response.data
    except ResponseError as e:
        return [{"error": f"Hotel List API Error: {e}"}]

    # --- Edge Case Handling for Phase 1 ---
    # If the Hotel List API returns no hotels, there is nothing to search for.
    if not hotels:
        return [{"message": f"No hotels found for city code: {city_code}"}]

    # Extract the hotel IDs from the response. The Hotel Search API requires these.
    hotel_ids = [hotel['hotelId'] for hotel in hotels]

    # --- Phase 2: Get Hotel Offers from Hotel IDs ---
    # This call uses the collected hotel IDs to find real-time offers.
    try:
        hotel_offers_response = amadeus.shopping.hotel_offers_search.get(
            hotelIds=','.join(hotel_ids[:20]),  # Using a slice to respect API limits
            adults=adults,
            checkInDate=checkin_date,
            checkOutDate=checkout_date,
            roomQuantity=1,
            bestRateOnly=True # This valid parameter can be used
        )
        offers = hotel_offers_response.data
    except ResponseError as e:
        # Handle API errors during the offer search.
        return

    # --- Edge Case Handling for Phase 2 ---
    # If hotels were found but no offers are available for the given dates.
    if not offers:
        return [{"message": f"No offers available for the hotels in {city_code} on the selected dates."}]

    return offers
# --------- WEATHER API ----------
# def get_weather_data(city):
#     url = "https://api.openweathermap.org/data/2.5/weather"
#     params = {
#         "q": city,
#         "appid": OPENWEATHER_API_KEY,
#         "units": "metric"
#     }
#     try:
#         res = requests.get(url, params=params)
#         res.raise_for_status()
#         return res.json()
#     except Exception as e:
#         return {"error": str(e)}

# --------- RESTAURANTS ----------
def get_city_coordinates(city):
    """
    Returns the latitude and longitude of a city using Nominatim.
    """
    geolocator = Nominatim(user_agent="restaurant_agent")
    location = geolocator.geocode(city)
    if location:
        return location.latitude, location.longitude
    else:
        raise ValueError(f"Could not geocode city: {city}")
    
def get_restaurants_data(lat, lon, radius=1000, limit=5):
    """
    Fetch nearby restaurants using Overpass API with coordinates.

    Args:
        lat (float): Latitude.
        lon (float): Longitude.
        radius (int): Radius in meters.
        limit (int): Number of restaurants.

    Returns:
        List of restaurants or error dict.
    """
    try:
        overpass_url = "http://overpass-api.de/api/interpreter"
        query = f"""
        [out:json];
        node
          ["amenity"="restaurant"]
          (around:{radius},{lat},{lon});
        out body;
        """
        res = requests.post(overpass_url, data={"data": query})
        res.raise_for_status()
        data = res.json()

        restaurants = []
        for element in data.get("elements", [])[:limit]:
            tags = element.get("tags", {})
            restaurants.append({
                "name": tags.get("name", "Unnamed Restaurant"),
                "cuisine": tags.get("cuisine", "Unknown"),
                "lat": element.get("lat"),
                "lon": element.get("lon"),
                "address": tags.get("addr:full") or f"{tags.get('addr:street', '')} {tags.get('addr:housenumber', '')}".strip()
            })

        return restaurants

    except Exception as e:
        return {"error": str(e)}

