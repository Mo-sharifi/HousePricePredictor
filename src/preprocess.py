import pandas as pd
import numpy as np

pd.options.display.float_format = "{:,.2f}".format
from tabulate import tabulate
from colorama import Fore, Style
from imblearn.over_sampling import RandomOverSampler
from data_loader import load_data


def remove_nan(df):
    """
    Remove rows with missing values from the DataFrame.
    Parameters:
    df (pd.DataFrame): The DataFrame to process.
    Returns:
    pd.DataFrame: The DataFrame with missing values removed.
    """
    # Remove rows with missing values
    
    df=df.dropna()

    return df 

def remove_doplicates(df):
    """
    Remove duplicate rows from the DataFrame.
    Parameters:
    df (pd.DataFrame): The DataFrame to process.
    Returns:
    pd.DataFrame: The DataFrame with duplicates removed.
    """
    # Remove duplicate rows
    data = df.drop_duplicates()
    return data

def remove_unnecessary_columns(df, columns):
    """
    Remove unnecessary columns from the DataFrame.
    Parameters:
    df (pd.DataFrame): The DataFrame to process.
    columns (list): List of column names to remove.
    Returns:
    pd.DataFrame: The DataFrame with unnecessary columns removed.
    """
    # Remove unnecessary columns
    df = df.drop(columns=columns)
    return df

def remove_outliers(df, column):
    """
    Remove outliers from a DataFrame based on the IQR method.

    Parameters:
    df (pd.DataFrame): The DataFrame to process.
    column (str): The column name to check for outliers.

    Returns:
    pd.DataFrame: The DataFrame with outliers removed.
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

def covert_to_numeric(df, columns):
    """
    Convert specified columns in a DataFrame to numeric type.

    Parameters:
    df (pd.DataFrame): The DataFrame to process.
    columns (list): List of column names to convert.

    Returns:
    pd.DataFrame: The DataFrame with specified columns converted to numeric.
    """
    for column in columns:
        if df.dtypes[column] == "bool":
            df[column] = df[column].astype("Int8")
        elif df[column].dtype == "object":
            df[column] = df[column].str.replace(",", "").astype(int)

    return df

def preprocess_data(df):
    """
    Preprocess the dataset by removing unnecessary columns, handling missing values,
    and converting data types.

    Parameters:
    df (pd.DataFrame): The DataFrame to preprocess.

    Returns:
    pd.DataFrame: The preprocessed DataFrame.
    """


    # Remove outliers
    df = remove_outliers(df, "Area")
    df = remove_outliers(df, "Price")

    return df

def oversample_binary_features(df, binary_features, target_ratio=0.4, random_state=42):
    """
    Oversamples the 0 values in each binary feature column to achieve the desired target ratio.

    Parameters:
        df (pandas.DataFrame): The input DataFrame containing the binary features.
        binary_features (list of str): List of binary feature column names, e.g., ['elevator', 'parking'].
        target_ratio (float, optional): The desired ratio of 0 values in the binary feature column. Default is 0.4 (40% of the values will be 0).
        random_state (int, optional): The seed used by the random number generator for reproducibility. Default is 42.

    Returns:
        pandas.DataFrame: A new DataFrame with oversampled 0 values in the specified binary feature columns.
    """
    
    df_result = df.copy()

    for feature in binary_features:
        X_temp = df_result.drop(columns=[feature])
        y_temp = df_result[feature]

        # نسبت فعلی 0
        count_0 = (y_temp == 0).sum()
        count_1 = (y_temp == 1).sum()
        current_ratio = count_0 / (count_0 + count_1)

        if current_ratio >= target_ratio:
            continue  # نیازی نیست

        # محاسبه نسبت موردنیاز برای imblearn
        sampling_strategy = target_ratio / (1 - target_ratio)

        ros = RandomOverSampler(sampling_strategy=sampling_strategy, random_state=random_state)
        X_resampled, y_resampled = ros.fit_resample(X_temp, y_temp)

        # ترکیب مجدد
        df_resampled = X_resampled.copy()
        df_resampled[feature] = y_resampled

        # آپدیت df_result
        df_result = df_resampled  # در هر مرحله دیتافریم بزرگ‌تر می‌شه

    return df_result


def log_dataframe(df, name="DataFrame", rows=5):
    df["Price"] = df["Price"].apply(lambda x: f"{x:,.0f}")
    print(Fore.CYAN + f"\n📊 Summary of {name}" + Style.RESET_ALL)
    print("-" * 50)

    # Shape
    print(f"🧩 Shape: {df.shape[0]} rows × {df.shape[1]} columns")

    # Column types
    print(f"\n🧬 Data types:")
    print(
        tabulate(
            df.dtypes.reset_index().rename(columns={"index": "Column", 0: "Type"}),
            headers="keys",
            tablefmt="grid",
        )
    )

    # Missing values
    missing = df.isnull().sum()
    if missing.sum() == 0:
        print(Fore.GREEN + "\n✅ No missing values." + Style.RESET_ALL)
    else:
        print(Fore.YELLOW + "\n⚠️ Missing values:" + Style.RESET_ALL)
        print(
            tabulate(
                missing[missing > 0]
                .reset_index()
                .rename(columns={"index": "Column", 0: "Missing"}),
                headers="keys",
                tablefmt="github",
            )
        )

    # Constant columns
    constant_cols = [col for col in df.columns if df[col].nunique() == 1]
    if constant_cols:
        print(
            Fore.MAGENTA
            + f"\nℹ️ Constant columns (only one unique value): {constant_cols}"
            + Style.RESET_ALL
        )

    # Sample preview
    print(f"\n🪞 Preview of first {rows} rows:")
    print(
        tabulate(df.head(rows), headers="keys", tablefmt="fancy_grid", showindex=False)
    )

    # Describe numerical columns
    print(
        Fore.GREEN + "\n📊 Summary statistics for numerical columns:" + Style.RESET_ALL
    )
    print(tabulate(df.describe().T, headers="keys", tablefmt="grid"))

    # Unique values for categorical columns
    cat_cols = ["Parking", "Warehouse", "Elevator"]
    print(Fore.BLUE + "\n🔎 Unique values in categorical columns:" + Style.RESET_ALL)
    for col in cat_cols:
        uniques = df[col].unique()
        print(f"  - {col}: {uniques[:10]}{' ...' if len(uniques) > 10 else ''}")

    print("-" * 50)


def main():
    data = load_data()
    data_not_nan = remove_nan(data)
    log_dataframe(data_not_nan, "Raw Data", 5)


if __name__ == "__main__":
    main()
