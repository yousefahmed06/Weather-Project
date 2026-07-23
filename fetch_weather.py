"""
Fetches 5-day / 3-hour forecast data for a list of cities from OpenWeatherMap
and appends each forecast interval as a row to weather_data.csv.

Each API call returns ~40 rows (5 days x 8 three-hour slots) per city, so
tracking 100 cities gives ~4,000 rows in a SINGLE run.

Run manually:
    OWM_API_KEY=your_key python fetch_weather.py

In GitHub Actions, OWM_API_KEY is injected from a repo secret (see the workflow file).
"""

import csv
import os
import time
from datetime import datetime, timezone

import requests

API_KEY = os.environ["OWM_API_KEY"]

# Edit this list to track whichever cities you want.
# "City,CountryCode" is used to avoid ambiguity for common city names
# (e.g. there are multiple "London"s and "Paris"s worldwide).
# More cities = more rows per run (each city adds ~40 rows) and more API calls.
CITIES = [
    # North America
    "New York,US", "Los Angeles,US", "Chicago,US", "Houston,US", "Miami,US",
    "San Francisco,US", "Seattle,US", "Boston,US", "Washington,US", "Atlanta,US",
    "Dallas,US", "Denver,US", "Toronto,CA", "Vancouver,CA", "Montreal,CA",
    "Calgary,CA", "Mexico City,MX",
    # South America
    "Sao Paulo,BR", "Rio de Janeiro,BR", "Buenos Aires,AR", "Lima,PE",
    "Bogota,CO",
    # Europe
    "London,GB", "Manchester,GB", "Edinburgh,GB", "Paris,FR", "Berlin,DE",
    "Munich,DE", "Frankfurt,DE", "Hamburg,DE", "Madrid,ES", "Barcelona,ES",
    "Rome,IT", "Milan,IT", "Amsterdam,NL", "Brussels,BE", "Vienna,AT",
    "Zurich,CH", "Geneva,CH", "Stockholm,SE", "Oslo,NO", "Copenhagen,DK",
    "Helsinki,FI", "Warsaw,PL", "Prague,CZ", "Budapest,HU", "Athens,GR",
    "Lisbon,PT", "Dublin,IE", "Moscow,RU", "Kiev,UA", "Bucharest,RO",
    # Middle East
    "Dubai,AE", "Abu Dhabi,AE", "Riyadh,SA", "Jeddah,SA", "Doha,QA",
    "Kuwait City,KW", "Tel Aviv,IL", "Amman,JO", "Beirut,LB", "Baghdad,IQ",
    # Africa
    "Cairo,EG", "Lagos,NG", "Nairobi,KE", "Johannesburg,ZA", "Cape Town,ZA",
    "Casablanca,MA", "Addis Ababa,ET",
    # South & Central Asia
    "Mumbai,IN", "Delhi,IN", "Bangalore,IN", "Chennai,IN", "Kolkata,IN",
    "Karachi,PK", "Lahore,PK", "Dhaka,BD", "Kathmandu,NP", "Colombo,LK",
    # East & Southeast Asia
    "Beijing,CN", "Shanghai,CN", "Hong Kong,HK", "Taipei,TW", "Tokyo,JP",
    "Osaka,JP", "Seoul,KR", "Bangkok,TH", "Singapore,SG", "Jakarta,ID",
    "Kuala Lumpur,MY", "Manila,PH", "Ho Chi Minh City,VN", "Hanoi,VN",
    "Istanbul,TR",
    # Oceania
    "Sydney,AU", "Melbourne,AU", "Brisbane,AU", "Perth,AU", "Auckland,NZ",
    "Wellington,NZ",
]

# De-duplicate while preserving order, in case of any accidental repeats above.
CITIES = list(dict.fromkeys(CITIES))

CSV_FILE = "weather_data.csv"

FIELDNAMES = [
    "fetch_timestamp",  # when this run pulled the data
    "city",
    "country",
    "forecast_datetime",  # the date/time this row's forecast applies to
    "temp_c",
    "feels_like_c",
    "temp_min_c",
    "temp_max_c",
    "pressure",
    "humidity",
    "wind_speed",
    "wind_deg",
    "wind_gust",
    "clouds_pct",
    "visibility",
    "pop",  # probability of precipitation, 0-1
    "rain_3h_mm",
    "snow_3h_mm",
    "weather_main",
    "weather_description",
    "part_of_day",  # d = day, n = night
]

# OpenWeatherMap's free tier allows 60 calls/minute. This delay keeps a
# 100-city run comfortably under that even with fast network responses.
REQUEST_DELAY_SECONDS = 1.1


def fetch_city_forecast(city: str) -> list[dict]:
    url = "https://api.openweathermap.org/data/2.5/forecast"
    params = {"q": city, "appid": API_KEY, "units": "metric"}

    response = requests.get(url, params=params, timeout=10)

    # Basic handling for rate-limit responses: wait and retry once.
    if response.status_code == 429:
        print(f"Rate limited on {city}, waiting 60s and retrying...")
        time.sleep(60)
        response = requests.get(url, params=params, timeout=10)

    response.raise_for_status()
    data = response.json()

    fetch_time = datetime.now(timezone.utc).isoformat()
    country = data.get("city", {}).get("country", "")

    rows = []
    for entry in data["list"]:
        main = entry["main"]
        wind = entry.get("wind", {})
        weather = entry["weather"][0]

        rows.append({
            "fetch_timestamp": fetch_time,
            "city": city,
            "country": country,
            "forecast_datetime": entry["dt_txt"],
            "temp_c": main["temp"],
            "feels_like_c": main["feels_like"],
            "temp_min_c": main["temp_min"],
            "temp_max_c": main["temp_max"],
            "pressure": main["pressure"],
            "humidity": main["humidity"],
            "wind_speed": wind.get("speed", ""),
            "wind_deg": wind.get("deg", ""),
            "wind_gust": wind.get("gust", ""),
            "clouds_pct": entry.get("clouds", {}).get("all", ""),
            "visibility": entry.get("visibility", ""),
            "pop": entry.get("pop", ""),
            "rain_3h_mm": entry.get("rain", {}).get("3h", 0),
            "snow_3h_mm": entry.get("snow", {}).get("3h", 0),
            "weather_main": weather["main"],
            "weather_description": weather["description"],
            "part_of_day": entry.get("sys", {}).get("pod", ""),
        })
    return rows


def main():
    file_exists = os.path.isfile(CSV_FILE)
    existing_keys = set()
    if file_exists:
        try:
            existing = pd.read_csv(
                CSV_FILE,
                  usecols=["city", "observation_datetime"]
            )

            existing_keys = set(
                zip(
                    existing["city"],
                    existing["observation_datetime"]
                )
            )

            print(f"Loaded {len(existing_keys)} existing records.")

        except Exception:
            pass
    

    with open(CSV_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if not file_exists:
            writer.writeheader()

        total = 0
        for i, city in enumerate(CITIES):
            try:
                row = fetch_city_weather(city, cc)
                key = (row["city"], row["observation_datetime"])
                if key not in existing_keys:
                    writer.writerow(row)
                    existing_keys.add(key)
                    total += 1
                    print( f"[{i}/{len(cities)}] " f"{city}, {cc} ✓ Added" )
                else:
                    print( f"[{i}/{len(cities)}] " f"{city}, {cc} ✓ Duplicate skipped")
            except Exception as e:
                # Don't let one failed city kill the whole run
                print(f"Failed for {city}: {e}")

            # Pace requests to stay under the free-tier rate limit,
            # skip the delay after the very last city.
            if i < len(CITIES) - 1:
                time.sleep(REQUEST_DELAY_SECONDS)

        print(f"Total cities attempted: {len(CITIES)}")
        print(f"Total rows written this run: {total}")


if __name__ == "__main__":
    main()
