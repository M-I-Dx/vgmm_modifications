import sys
import os

# Add the parent directory of 'src' to the Python module search path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import yaml
from src.data_loader import DataLoaderTransformer


def load_config(config_path):
    """
    Load the configuration file from a YAML path.

    Parameters:
        config_path (str): Path to the YAML configuration file.

    Returns:
        dict: Parsed configuration dictionary.
    """
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config


def save_output(df, output_dir, filename_base):
    """
    Save a Dask DataFrame in both CSV and Parquet formats in a specified directory.

    Parameters:
        df (dd.DataFrame): Dask DataFrame to save.
        output_dir (str): Directory where to save the output files.
        filename_base (str): Base name of the output files (without extension).
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Save to CSV
    csv_path = os.path.join(output_dir, f"{filename_base}.csv")
    df.compute().to_csv(csv_path, index=False)
    print(f"Saved CSV to {csv_path}")

    # Save to Parquet
    parquet_path = os.path.join(output_dir, f"{filename_base}.parquet")
    df.to_parquet(parquet_path, engine="pyarrow", write_index=False)
    print(f"Saved Parquet to {parquet_path}")


def main(config_path):
    # Load configuration
    config = load_config(config_path)
    data_loader_config = config["DataLoaderTransformer"]
    vgmm_config = config["ScalableVGMMNormalizer"]

    # Initialize DataLoaderTransformer
    data_loader = DataLoaderTransformer(config=data_loader_config)

    # Load and preprocess data
    df = data_loader.load_and_convert_data()
    df = data_loader.apply_transformations(df)

    # Define continuous columns for transformation
    continuous_columns = config.get("continuous_columns", ["col1", "col2", "col3"])

    # Apply VGMM Transformation
    transformed_df, normalizers = data_loader.apply_vgmm_transformation(df, continuous_columns)

    # Extract only the transformed columns (excluding probability columns)
    transformed_only = transformed_df[[f"{col}_transformed" for col in continuous_columns]]

    # Define output directory path based on the input path
    input_dir = os.path.dirname(data_loader_config["input_path"])
    output_dir = os.path.join(input_dir, "output")

    # Save transformed data
    save_output(transformed_df, output_dir, "transformed_data_complete")  
    save_output(transformed_only, output_dir, "transformed_data")

    # Reload transformed data with additional columns for inverse transformation
    # Note: These columns are necessary for inverse transformation
    transformed_full_df = transformed_df

    # Apply inverse transformation
    inverse_transformed_df = data_loader.apply_inverse_transformation(
        transformed_full_df, continuous_columns, normalizers
    )

    # Extract only the inverse-transformed columns
    inverse_transformed_only = inverse_transformed_df[[f"{col}_inverse_transformed" for col in continuous_columns]]

    # Save inverse-transformed data
    save_output(inverse_transformed_only, output_dir, "inverse_transformed_data")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python main.py <config_path>")
        sys.exit(1)

    config_path = sys.argv[1]
    main(config_path)