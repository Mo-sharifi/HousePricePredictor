import pandas as pd


def load_tehran_data(
    file_path="/home/dili/Univers/HousePricePredictor/data/kashefi_dataset.csv",
) -> pd.DataFrame:
    """
    Load data from a CSV file.

    Parameters:
    file_path (str): The path to the CSV file.

    Returns:
    pd.DataFrame: The loaded data as a pandas DataFrame.

    """

    try:
        data = pd.read_csv(file_path)

        return data

    except FileNotFoundError:

        print(f"Error: The file {file_path} was not found.")

        return None

    except pd.errors.EmptyDataError:
        print(f"Error: The file {file_path} is empty.")

        return None

    finally:

        print("Data loading process completed.")


def load_geolocation_data(
    file_path=r"/home/dili/Univers/HousePricePredictor/data/lat_long_address.csv",
):
    """
    Load geolocation data from a CSV file.

    Parameters:
    file_path (str): The path to the CSV file.

    Returns:
    pd.DataFrame: The loaded geolocation data as a pandas DataFrame.
    """

    try:
        data = pd.read_csv(file_path)
        data.rename(columns={"place": "Address"}, inplace=True)

        return data

    except FileNotFoundError:

        print(f"Error: The file {file_path} was not found.")

        return None

    except pd.errors.EmptyDataError:
        print(f"Error: The file {file_path} is empty.")

        return None

    finally:

        print("Geolocation data loading process completed.")


def main():

    data = load_tehran_data()
    df_geolocation = load_geolocation_data()

    from preprocess import log_dataframe

    if data is not None:
        log_dataframe(data, "Data", 5)
    if df_geolocation is not None:
        log_dataframe(df_geolocation, "Geolocation Data", 5)


if __name__ == "__main__":

    main()
