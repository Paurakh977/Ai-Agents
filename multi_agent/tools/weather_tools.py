import requests

def get_weather_by_country(country: str) -> dict:
    """
    Fetches current temperature (°C) and wind speed (km/h) given a country name.
    Uses Open-Meteo's free Geocoding + Weather APIs — no API key required.
    
    Returns:
      {
        "success": bool,
        "country": str,
        "latitude": float|null,
        "longitude": float|null,
        "temperature_c": float|null,
        "wind_speed_kmh": float|null,
        "error": str|null
      }
    """
    try:
        # Step 1: Look up coordinates via Open-Meteo geocoding
        geo_url = "https://geocoding-api.open-meteo.com/v1/search"
        geo_params = {"name": country, "count": 1, "language": "en", "format": "json"}
        geo_r = requests.get(geo_url, params=geo_params, timeout=5)
        geo_r.raise_for_status()
        geo_data = geo_r.json()
        results = geo_data.get("results")
        if not results:
            raise Exception(f"No location found for country '{country}'")
        loc = results[0]
        lat = loc["latitude"]
        lon = loc["longitude"]

        # Step 2: Fetch current weather
        weather_url = "https://api.open-meteo.com/v1/forecast"
        weather_params = {
            "latitude": lat,
            "longitude": lon,
            "current_weather": "true"
        }
        weather_r = requests.get(weather_url, params=weather_params, timeout=5)
        weather_r.raise_for_status()
        cw = weather_r.json().get("current_weather")
        if not cw:
            raise Exception("No current weather data available")

        return {
            "success": True,
            "country": country,
            "latitude": lat,
            "longitude": lon,
            "temperature_c": cw.get("temperature"),
            "wind_speed_kmh": cw.get("windspeed"),
            "error": None
        }

    except Exception as e:
        return {
            "success": False,
            "country": country,
            "latitude": None,
            "longitude": None,
            "temperature_c": None,
            "wind_speed_kmh": None,
            "error": str(e)
        }


