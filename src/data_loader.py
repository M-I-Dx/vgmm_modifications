from dask.distributed import Client
import dask.dataframe as dd
import numpy as np
from src.vgmm_normalizer import ScalableVGMMNormalizer


class DataLoaderTransformer:
    def __init__(self, config=None):
        """
        Initialize the DataLoaderTransformer class with a configured Dask client.

        Parameters:
            config (dict): Configuration dictionary for parameters and paths.
        """
        # Default configuration
        default_config = {
            "input_path": None,
            "output_parquet_path": None,
            "n_workers": 1,
            "threads_per_worker": 2,
            "memory_limit": "6GB",
            "spill_dir": "/tmp/dask-spill",
            "target_partitions": 10,
            "sample_fraction": 0.01,
            "random_state": 42,
            "n_clusters": 10,
            "eps": 0.005,
        }

        # Update default configuration with user-provided values
        self.config = {**default_config, **(config or {})}

        self.input_path = self.config["input_path"]
        self.output_parquet_path = self.config["output_parquet_path"]
        self.target_partitions = self.config["target_partitions"]

        # Initialize Dask client with optimized settings
        self.client = Client(
            n_workers=self.config["n_workers"],
            threads_per_worker=self.config["threads_per_worker"],
            memory_limit=self.config["memory_limit"],
            local_directory=self.config["spill_dir"],
        )
        print(
            f"Dask client initialized with {self.config['n_workers']} workers, "
            f"{self.config['threads_per_worker']} threads per worker, memory limit {self.config['memory_limit']}, "
            f"and spill-to-disk at {self.config['spill_dir']}."
        )

    def load_and_convert_data(self) -> dd.DataFrame:
        """
        Load data from CSV or Parquet, convert CSV to Parquet if necessary, and repartition.

        Returns:
            dd.DataFrame: Loaded data in a Dask DataFrame with desired partitions.
        """
        if not self.input_path:
            raise ValueError("Input path is not specified in the configuration.")

        if self.input_path.endswith(".csv"):
            df = dd.read_csv(self.input_path)
            df = df.repartition(npartitions=self.target_partitions)
            df.to_parquet(self.output_parquet_path, engine="pyarrow", write_index=False)
            print(f"CSV converted to Parquet and saved at {self.output_parquet_path}")
            df = dd.read_parquet(self.output_parquet_path)
        elif self.input_path.endswith(".parquet"):
            df = dd.read_parquet(self.input_path)
            print("Parquet file loaded directly for processing.")
        else:
            raise ValueError(
                "Unsupported file format. Please provide a CSV or Parquet file."
            )
        return df

    def apply_transformations(self, df: dd.DataFrame) -> dd.DataFrame:
        """
        Apply initial transformations, such as handling missing values.

        Parameters:
            df (dd.DataFrame): Input DataFrame to transform.

        Returns:
            dd.DataFrame: Transformed DataFrame.
        """
        df = df.dropna()  # Example transformation to drop nulls
        print("Initial transformations applied to the DataFrame.")
        return df

    def apply_vgmm_transformation(self, df: dd.DataFrame, continuous_columns):
        """
        Apply VGMM normalization on continuous columns using ScalableVGMMNormalizer.

        Parameters:
            df (dd.DataFrame): The Dask DataFrame containing continuous columns to normalize.
            continuous_columns (list): List of continuous column names to normalize.

        Returns:
            Tuple[dd.DataFrame, dict]: Transformed DataFrame and the dictionary of fitted normalizers.
        """
        # Fit the model on a small sampled subset
        sample_df = (
            df[continuous_columns]
            .sample(
                frac=self.config["sample_fraction"],
                random_state=self.config["random_state"],
            )
            .compute()
        )

        normalizers = {
            col: ScalableVGMMNormalizer(
                config={
                    "n_clusters": self.config["n_clusters"],
                    "eps": self.config["eps"],
                }
            )
            for col in continuous_columns
        }

        for col in continuous_columns:
            normalizers[col].fit(sample_df[col].values)

        # Transform the dataset in chunks
        for col in continuous_columns:
            transformed_data = df.map_partitions(
                lambda part: normalizers[col].transform_chunk(part[col].values),
                meta=(col, "object"),
            )

            # Extract the transformed values (1st column of the output) as the main transformed column
            df[f"{col}_transformed"] = transformed_data.map_partitions(
                lambda arr: arr[:, 0], meta=(f"{col}_transformed", "f8")
            )

            # Dynamically handle only the active components for probability columns
            active_components = sum(
                normalizers[col].components
            )  # Number of active components
            for i in range(active_components):
                df[f"{col}_prob_{i}"] = transformed_data.map_partitions(
                    lambda arr, idx=i: arr[:, idx + 1], meta=(f"{col}_prob_{i}", "f8")
                )

        return df, normalizers

    def apply_inverse_transformation(
        self, df: dd.DataFrame, continuous_columns, normalizers
    ):
        """
        Apply inverse VGMM transformation using ScalableVGMMNormalizer.

        Parameters:
            df (dd.DataFrame): The Dask DataFrame containing transformed data.
            continuous_columns (list): List of continuous column names to inverse transform.
            normalizers (dict): Dictionary of fitted ScalableVGMMNormalizer objects keyed by column names.

        Returns:
            dd.DataFrame: DataFrame with inverse-transformed columns.
        """
        for col in continuous_columns:
            transformed_col = f"{col}_transformed"
            if transformed_col not in df.columns:
                raise ValueError(
                    f"Transformed column '{transformed_col}' not found in the DataFrame."
                )

            # Combine transformed values and probability columns into the expected format
            prob_cols = [
                f"{col}_prob_{i}"
                for i in range(normalizers[col].n_clusters)
                if f"{col}_prob_{i}" in df.columns
            ]
            if not prob_cols:
                raise ValueError(f"No probability columns found for '{col}'.")

            # Create a multi-dimensional array by concatenating transformed values and probabilities
            def combine_columns(partition):
                transformed_values = partition[transformed_col].values.reshape(-1, 1)
                probabilities = partition[prob_cols].values
                return np.hstack([transformed_values, probabilities])

            combined = df.map_partitions(
                combine_columns, meta=(f"{col}_combined", "object")
            )

            # Perform inverse transformation
            df[f"{col}_inverse_transformed"] = combined.map_partitions(
                lambda part: normalizers[col].inverse_transform_chunk(part),
                meta=(f"{col}_inverse_transformed", "f8"),
            )

        return df
