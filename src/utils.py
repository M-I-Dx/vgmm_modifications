from sklearn.metrics import mean_absolute_error, mean_squared_error
import pandas as pd


def calculate_metrics(original_path, transformed_path, continuous_columns):
    """
    Calculate evaluation metrics (MAE, RMSE) between original and inverse-transformed data.

    Parameters:
        original_path (str): Path to the CSV file containing the original data.
        transformed_path (str): Path to the CSV file containing the transformed and inverse-transformed data.
        continuous_columns (list of str): List of continuous column names to evaluate.

    Returns:
        dict: A dictionary where keys are column names and values are dictionaries with MAE and RMSE metrics.
              Example:
              {
                  "column1": {"MAE": 0.123, "RMSE": 0.456},
                  "column2": {"MAE": 0.789, "RMSE": 0.012}
              }

    Raises:
        ValueError: If any column specified in `continuous_columns` is not found in the provided files.

    Example Usage:
        metrics = calculate_metrics(
            original_path="original_data.csv",
            transformed_path="transformed_data.csv",
            continuous_columns=["col1", "col2"]
        )
        print(metrics)
    """
    # Read the original and transformed datasets
    original_data = pd.read_csv(original_path)
    transformed_data = pd.read_csv(transformed_path)

    # Initialize the results dictionary
    results = {}

    for col in continuous_columns:
        if col not in original_data.columns:
            raise ValueError(f"Column '{col}' not found in the original data.")
        if f"{col}_inverse_transformed" not in transformed_data.columns:
            raise ValueError(
                f"Column '{col}_inverse_transformed' not found in the transformed data."
            )

        # Compute metrics
        mae = mean_absolute_error(
            original_data[col], transformed_data[f"{col}_inverse_transformed"]
        )
        mse = mean_squared_error(
            original_data[col], transformed_data[f"{col}_inverse_transformed"]
        )
        rmse = mse**0.5

        # Store metrics in the results dictionary
        results[col] = {"MAE": mae, "RMSE": rmse}

    return results


def validate_file_format(file_path, allowed_extensions=("csv", "parquet")):
    """
    Validate that a file has an allowed extension.

    Parameters:
        file_path (str): Path to the file to validate.
        allowed_extensions (tuple of str): Tuple of allowed file extensions. Defaults to ("csv", "parquet").

    Returns:
        None

    Raises:
        ValueError: If the file's extension is not in the allowed extensions.

    Example Usage:
        try:
            validate_file_format("data.csv")
            print("File format is valid.")
        except ValueError as e:
            print(e)
    """
    # Check if the file extension is in the allowed list
    if not file_path.endswith(allowed_extensions):
        raise ValueError(
            f"Invalid file format. Expected one of {allowed_extensions}. Got: {file_path}"
        )
