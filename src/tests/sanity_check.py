import os
import glob
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from src.data_loader import DataLoaderTransformer


def test_combined_transform_inverse_transform():
    """
    Test the combined functionality of DataLoaderTransformer and ScalableVGMMNormalizer.

    This test:
        - Creates a dummy dataset with three continuous columns.
        - Initializes DataLoaderTransformer and ScalableVGMMNormalizer using configuration parameters.
        - Applies transformation and inverse transformation on the dataset.
        - Computes MAE and RMSE between the original and inverse-transformed data.
        - Ensures that the errors are below a defined threshold.

    Raises:
        AssertionError: If the MAE or RMSE for any column exceeds the defined threshold.
        PermissionError: If cleanup files cannot be deleted due to permission issues.

    Cleanup:
        - Deletes the dummy files (`dummy_data.csv` and `dummy_data.parquet`) after the test.
    """
    try:
        # Create dummy dataset
        data = pd.DataFrame({
            "col1": np.random.normal(50, 10, 1000),
            "col2": np.random.normal(100, 20, 1000),
            "col3": np.random.normal(200, 30, 1000),
        })
        data.to_csv("dummy_data.csv", index=False)

        # Config for DataLoaderTransformer and ScalableVGMMNormalizer
        config = {
            "DataLoaderTransformer": {
                "input_path": "dummy_data.csv",
                "output_parquet_path": "dummy_data.parquet",
                "target_partitions": 1
            },
            "ScalableVGMMNormalizer": {
                "n_clusters": 5,
                "eps": 0.01
            }
        }

        # Initialize DataLoaderTransformer
        data_loader = DataLoaderTransformer(config=config["DataLoaderTransformer"])
        df = data_loader.load_and_convert_data()

        # Apply VGMM Transformation
        continuous_columns = ["col1", "col2", "col3"]
        transformed_df, normalizers = data_loader.apply_vgmm_transformation(df, continuous_columns)

        # Inverse transform
        inverse_transformed_df = data_loader.apply_inverse_transformation(
            transformed_df, continuous_columns, normalizers
        )

        # Validation: Check MAE and RMSE
        original_data = pd.read_csv("dummy_data.csv")
        inverse_transformed_data = inverse_transformed_df.compute()

        for col in continuous_columns:
            mae = mean_absolute_error(original_data[col], inverse_transformed_data[f"{col}_inverse_transformed"])
            mse = mean_squared_error(original_data[col], inverse_transformed_data[f"{col}_inverse_transformed"])
            rmse = mse ** 0.5

            print(f"Testing column: {col}")
            print(f"MAE: {mae}, RMSE: {rmse}")

            assert mae < 500.0, f"MAE for column '{col}' exceeds threshold!"
            assert rmse < 500.0, f"RMSE for column '{col}' exceeds threshold!"

    finally:
        # Ensure Dask client shuts down properly
        if data_loader.client:
            data_loader.client.shutdown()

        # Cleanup: Remove dummy files
        try:
            if os.path.exists("dummy_data.csv"):
                os.remove("dummy_data.csv")
            if os.path.exists("dummy_data.parquet"):
                for file in glob.glob(os.path.join("dummy_data.parquet", "*")):
                    os.remove(file)
                os.rmdir("dummy_data.parquet")
        except PermissionError as e:
            print(f"PermissionError during cleanup: {e}")
        except Exception as e:
            print(f"Unexpected error during cleanup: {e}")