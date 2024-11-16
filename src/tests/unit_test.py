import os
import pandas as pd
from glob import glob
from src.vgmm_normalizer import ScalableVGMMNormalizer
from src.data_loader import DataLoaderTransformer


def test_scalable_vgmm_normalizer_initialization():
    """
    Test initialization and setup of ScalableVGMMNormalizer.

    This test:
        - Initializes ScalableVGMMNormalizer with specified configuration.
        - Verifies that all attributes are correctly initialized before fitting.

    Raises:
        AssertionError: If any attribute is not initialized as expected.
    """
    # Config for ScalableVGMMNormalizer
    config = {
        "n_clusters": 3,
        "eps": 0.01
    }

    # Initialize the normalizer
    normalizer = ScalableVGMMNormalizer(config=config)

    # Check that parameters are set correctly
    assert normalizer.n_clusters == config["n_clusters"], "Incorrect number of clusters!"
    assert normalizer.eps == config["eps"], "Epsilon value not set correctly!"
    assert normalizer.model is None, "Model should be None before fitting!"
    assert normalizer.components == [], "Components should be empty before fitting!"
    assert normalizer.ordering is None, "Ordering should be None before fitting!"


def test_data_loader_transformer_initialization_and_partitions():
    """
    Test initialization and setup of DataLoaderTransformer, and verify the number of partition files.

    This test:
        - Initializes DataLoaderTransformer with a dummy dataset.
        - Ensures that the configuration is applied correctly.
        - Verifies that the output Parquet file contains the correct number of partitions.

    Raises:
        AssertionError: If any configuration parameter is not set correctly or the number of partitions is incorrect.
        PermissionError: If cleanup files cannot be deleted due to permission issues.

    Cleanup:
        - Deletes the dummy files (`dummy_data.csv` and `dummy_data.parquet`) after the test.
    """
    try:
        # Create dummy dataset
        data = pd.DataFrame({
            "col1": [1, 2, 3, 4, 5],
            "col2": [10, 20, 30, 40, 50],
        })
        data.to_csv("dummy_data.csv", index=False)

        # Config for DataLoaderTransformer
        config = {
            "input_path": "dummy_data.csv",
            "output_parquet_path": "dummy_data.parquet",
            "n_workers": 2,
            "threads_per_worker": 1,
            "memory_limit": "2GB",
            "spill_dir": "/tmp/dask-spill",
            "target_partitions": 3  # Number of partitions to test
        }

        # Initialize DataLoaderTransformer
        data_loader = DataLoaderTransformer(config=config)

        # Check configurations
        assert data_loader.input_path == config["input_path"], "Input path not set correctly!"
        assert data_loader.output_parquet_path == config["output_parquet_path"], "Output path not set correctly!"
        assert data_loader.target_partitions == config["target_partitions"], "Target partitions not set correctly!"

        # Ensure Dask client is initialized
        assert data_loader.client is not None, "Dask client not initialized!"

        # Load and convert data
        df = data_loader.load_and_convert_data()

        # Verify number of partition files
        parquet_files = glob(os.path.join(config["output_parquet_path"], "*.parquet"))
        assert len(parquet_files) == config["target_partitions"], (
            f"Expected {config['target_partitions']} partition files, but found {len(parquet_files)}!"
        )
    finally:
        # Ensure Dask client shuts down properly
        if data_loader.client:
            data_loader.client.shutdown()
        
        # Attempt cleanup
        try:
            if os.path.exists("dummy_data.csv"):
                os.remove("dummy_data.csv")
            if os.path.exists("dummy_data.parquet"):
                for file in glob(os.path.join("dummy_data.parquet", "*")):
                    os.remove(file)
                os.rmdir("dummy_data.parquet")
        except PermissionError as e:
            print(f"PermissionError during cleanup: {e}")