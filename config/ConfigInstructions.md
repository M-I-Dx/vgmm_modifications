# README for Config File

This document explains the parameters in the `config.yaml` file and their respective roles in configuring the behavior of the system for transforming and normalizing continuous data using the Variational Gaussian Mixture Model (VGMM).

---

## **1. DataLoaderTransformer Section**
This section configures the data loading and chunk-wise transformation process.

- **`input_path`**  
  Specifies the path to the input dataset in CSV format.  
  *Example:* `./data/input_50K.csv`

- **`output_parquet_path`**  
  Defines the path where the input csv file will be saved in Parquet format with the specified number of partitions for future further use. 
  *Example:* `./data/input_converted_50K.parquet`

- **`n_workers`**  
  Sets the number of Dask workers for parallel processing.  
  *Default:* `1`  

- **`threads_per_worker`**  
  Specifies the number of threads each worker should use.  
  *Default:* `2`

- **`memory_limit`**  
  Allocates a memory cap for each worker to prevent overload. Use standard memory size units like `GB` or `MB`.  
  *Example:* `6GB`

- **`spill_dir`**  
  Directory for storing temporary files when the memory exceeds the limit (spilling).  
  *Example:* `./tmp/dask-spill`

- **`target_partitions`**  
  Number of partitions to split the dataset into for processing. Higher values improve memory efficiency at the cost of processing speed.  
  *Default:* `10`

---

## **2. ScalableVGMMNormalizer Section**
This section configures the Variational Gaussian Mixture Model (VGMM) parameters for data normalization.

- **`n_clusters`**  
  Specifies the maximum number of mixture components for the VGMM. The effective number of components can be smaller, as components with negligible weights (close to zero) are excluded during model fitting.  
  *Default:* `10`

- **`eps`**  
The `eps` parameter defines the minimum weight threshold that a component in the VGMM must exceed to be considered "significant." Components with weights below this threshold are excluded from further processing, ensuring that only meaningful clusters are retained.
  *Default:* `0.005`

- **`weight_concentration_prior_type`**  
 Determines the type of weight concentration prior for the model.  
  *Options:*  
    - `dirichlet_process` (default): Suitable for flexible, non-fixed number of clusters. If you have a dataset where you are unsure how many clusters are meaningful, use `dirichlet_process`. The model will figure it out for you.   
    - `dirichlet_distribution`: For a fixed number of clusters. If you know you want to use exactly x clusters, use `dirichlet_distribution`.

- **`weight_concentration_prior`**  
 The Dirichlet concentration parameter (commonly referred to as “gamma” in the literature) governs the weight distribution across components in the model. A higher concentration focuses more weight near the center, resulting in more active components. Conversely, a lower concentration shifts weight toward the edges of the mixture weights simplex, promoting sparsity and reducing the number of active components..  
  *Default:* `0.001`

- **`max_iter`**  
  Maximum number of iterations for the VGMM fitting process. More iterations improve model accuracy at the cost of time.  
  *Default:* `100`

- **`n_init`**  
 Specifies the number of times the model should run with different random initializations. The result with the best likelihood is chosen. This helps avoid getting stuck in a "bad" initialization. Running more initializations increases computation time but improves robustness.
  *Default:* `1`

- **`random_state`**  
  Seed for random number generation to ensure reproducibility.  
  *Example:* `42`

---

## **3. Continuous Columns Section**
This section lists the columns in the dataset that contain continuous values to be transformed by the VGMM.

- **`continuous_columns`**  
  A list of continuous column names to be normalized.  
  *Example:*  
  ```yaml
  continuous_columns:
    - "Col1"
    - "Col2"
    - "Col3"