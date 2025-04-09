import requests
import pandas as pd
import time
from data_loader import load_tehran_data

from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("API_KEY")


def get_all_addresses():
    """
    Extracts a list of non-null addresses from the loaded dataset.

    Returns:
        list[str]: A list of addresses from the 'Address' column.
                   Returns an empty list if the column is not found or data is missing.
    """

    data = load_tehran_data()
    if data is not None and "Address" in data.columns:
        return data["Address"].dropna().tolist()
    else:
        print("❌ Error: 'Address' column not found in the DataFrame.")
        return []


def get_coordinates(place_name, api_key=api_key):
    """
    Retrieves the geographic coordinates (latitude and longitude) for a given address
    using the OpenCage Geocoding API.

    Args:
        place_name (str): The address or place name to geocode.
        api_key (str): The API key for accessing the OpenCage API.

    Returns:
        tuple[float, float] | tuple[None, None]:
            A tuple containing latitude and longitude.
            Returns (None, None) if the request fails or no result is found.

    """

    try:
        query = f"{place_name}, Tehran, Iran"
        url = f"https://api.opencagedata.com/geocode/v1/json?q={query}&key={api_key}&language=fa&countrycode=ir"

        r = requests.get(url)
        if r.status_code == 200:
            result = r.json()
            if result["results"]:
                geometry = result["results"][0]["geometry"]
                return geometry["lat"], geometry["lng"]
        return None, None
    except Exception as e:
        print(f"Error for '{place_name}': {e}")
        return None, None


def generate_coordinates(address_list, delay=1):
    """
    Takes a list of addresses and returns a DataFrame with their corresponding
    latitude and longitude, using caching to avoid redundant API calls.

    Args:
        address_list (list[str]): A list of addresses to geocode.
        delay (int): Delay (in seconds) between API calls to avoid rate limits (default is 1).

    Returns:
        pd.DataFrame: A DataFrame containing the original address list along with
                      their corresponding latitude and longitude columns.
    """

    unique_places = list(set(address_list))
    cache = {}

    for idx, place in enumerate(unique_places):
        lat, lon = get_coordinates(place)
        print(f"{idx + 1}/{len(unique_places)} → {place}: {lat}, {lon}")
        cache[place] = {"latitude": lat, "longitude": lon}
        time.sleep(delay)  # to avoid hitting the API rate limit

    # Create a DataFrame with the original address list and their coordinates

    data = []
    for place in address_list:
        coords = cache.get(place, {"latitude": None, "longitude": None})
        data.append(
            {
                "Address": place,
                "Latitude": coords["latitude"],
                "Longitude": coords["longitude"],
            }
        )

    return pd.DataFrame(data)


def save_coordinates_to_csv(
    output_path=r"/home/dili/Univers/HousePricePredictor/data/lat_long_address.csv",
):
    """
    Main wrapper function that loads addresses, generates coordinates, and saves
    the result to a CSV file.

    Args:
        output_path (str): File path to save the final DataFrame (default: "../data/lat_long_address.csv").

    Returns:
        None
    """

    addresses = get_all_addresses()
    if not addresses:
        print("No addresses found to geocode.")
        return

    df = generate_coordinates(addresses)
    df.to_csv(output_path, index=False)
    print(f"saved successfully on : {output_path} ")


def main():
    """
    Entry point for the geolocation script.
    Calls the function to process addresses and save coordinates to a CSV file.
    """

    save_coordinates_to_csv()


if __name__ == "__main__":
    main()
