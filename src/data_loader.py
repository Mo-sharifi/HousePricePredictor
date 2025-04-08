import pandas as pd


def load_data(
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


def main():
    data = load_data()

    print(data.head())


if __name__ == "__main__":

    main()
