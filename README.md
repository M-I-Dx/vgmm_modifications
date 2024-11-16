# VGMM Modifications

## Task Overview
This take-home assignment focused on implementing a data transformation technique to enhance the handling of continuous data in a generative model. The core approach involved training a Bayesian Gaussian Mixture Model (BGMM) on each continuous column in the dataset. The trained model was then used to transform the data, making it more suitable for generative processing and improving the model’s ability to handle continuous variables effectively.

The main challenge with this approach lies in its lack of scalability. The assignment’s goal was to address this limitation and make the technique scalable enough to handle datasets containing up to 1 billion rows. Two primary issues contribute to the scalability problem:

**Training Limitations of the Bayesian Gaussian Mixture Model:**
The current implementation relies on the sklearn library for training the Bayesian Gaussian Mixture Model (BGMM). After reviewing the mathematical foundation of the model and its implementation in sklearn, I observed that stochastic training using minibatches is not feasible. Consequently, training the model requires loading the entire dataset into memory, which is impractical for very large datasets. This necessitates finding an alternative approach for training the model.

**Scaling the Transformation Process:**
Once the model is trained, the next step involves applying the transformation to the entire dataset. For massive datasets, this requires efficient utilization of the system’s multiprocessing capabilities. Additionally, a scalable framework needs to be established to support multi-core, multi-device systems for handling such extensive transformations.

## My thought process

#### Optimizing the training process of Gaussian Mixture Model
For this task, I considered two potential solutions to address the challenge of training a Bayesian Gaussian Mixture Model (BGMM) on massive datasets.

**1. Exploring Alternative Models and Libraries**
One approach involved finding an alternative to the BGMM that could achieve similar transformations but be trained on smaller subsets of data. During my research, I came across an insightful blog post titled _Bayesian Gaussian Mixture Modeling with Stochastic Variational Inference_ [1]. This post demonstrated how the TensorFlow Probability library could be used to train a BGMM on minibatches of data.
However, this implementation assumed that the number of modes in the BGMM was predefined and fixed. This approach differs from the method outlined in the CTGAN paper, where the optimal number of modes is determined dynamically based on the data. While I did not pursue this implementation due to this limitation, it represents a promising avenue for further research. Libraries like TensorFlow Probability and PyMC specialize in training probabilistic models, offering broader capabilities and flexibility. Investigating stochastic training methods that determine the number of modes dynamically could address the scalability issue effectively.

**2. Simplified Subset-Based Training**
The second approach, though simpler, was less robust and carried significant limitations. Due to the time constraints of this assignment, I opted for this method. It involved randomly selecting a subset of the dataset for training the BGMM. The rationale was that, assuming the dataset had an unbiased distribution, a random subset would provide a representative sample for training. The trained model was then used to perform the transformation and validate it using a sanity check with the inverse transformation.
While this approach is straightforward and avoids the memory bottleneck of loading the entire dataset, it has several downsides:
1. **Loss of Representation:** If the dataset contains rare patterns or outliers, random sampling might fail to capture them, resulting in a poorly generalized model.
2. **Validation Limitations:** The sanity check, which ensures that transformations and inverse transformations align, is insufficient to fully validate this approach’s efficacy, especially without integrating it into the generative model.

Despite its simplicity, this method serves as a stopgap solution. Further testing and integration with the generative model are necessary to assess its practical utility

#### Scaling the Transformation process and other computations 
Once the BGMM models are trained, they must be used to transform the dataset in preparation for training generative models. Given the assumption of access to workstation-grade computational resources with multi-core and multi-machine capabilities, it is essential to design a scalable system that efficiently utilizes these resources.
To achieve this, I evaluated two main frameworks: **Apache Spark** and **Dask**.

**Exploring Apache Spark**
I began with Apache Spark, as it is a well-established framework in the field of big data processing, offering robust community support and extensive resources for implementation. Spark’s reputation as a mature technology made it an appealing starting point. However, upon a deeper exploration, I encountered several significant limitations:
- **Limited MLlib Capabilities:** Spark’s MLlib library lacked an implementation of the `BayesianGaussianMixture` model found in the sklearn library. This absence meant that core functionality required for my task would need to be built from scratch.
- **Insufficient Metrics Support:** The MLlib library also lacked support for many metrics critical for evaluating tabular generative models. Incorporating these metrics would necessitate creating numerous User Defined Functions (UDFs), which are known to degrade Spark’s performance significantly.
- **Compatibility Challenges:** Spark is primarily designed for scalability and speed across distributed systems but is not natively aligned with Python. Translating research code that leverages Python-centric libraries such as NumPy and sklearn into production code compatible with Spark would require extensive rewrites. This additional overhead undermines the flexibility and efficiency of the research-to-production pipeline.
Given these challenges, I decided against pursuing Spark further for this implementation.

**Exploring Dask**
Next, I shifted my focus to **Dask**. The primary appeal of this library lies in its native integration with Python and its seamless compatibility with widely used Python libraries such as NumPy, Pandas, and sklearn. Unlike Apache Spark, Dask is designed to work within the Python ecosystem, which makes it an ideal choice for translating research code into production-ready implementations.

Dask also supports scalability across multi-node clusters, enabling it to handle large datasets effectively. While it may lack the cohesiveness and extensive ecosystem of Spark, Dask compensates for this with its flexibility and ease of integration into Python workflows. This characteristic significantly reduces the overhead for researchers, allowing them to focus more on experimentation and less on re-engineering their codebase for production environments.

This compatibility and flexibility make Dask particularly attractive for projects that require iterative development and close alignment with research pipelines. It minimizes the need for extensive rewrites, thereby streamlining the transition from research prototypes to scalable production systems.

In the following sections, I will detail how I integrated Dask into my implementation and leveraged its capabilities to address the challenges of scaling transformations and computations.

Note: I also considered the idea of chunking the data using libraries such as **Polars** or **Pandas** and leveraging the vectorization capabilities of NumPy for the transformation process. This approach can be seen as a simplified version of Dask that eliminates the need to learn a new framework. By processing data in chunks, it avoids the memory limitations of working with large datasets in a single load while maintaining the speed benefits of NumPy’s vectorized operations.
However, this approach has a significant limitation: it cannot scale beyond a single machine. While it might be suitable for smaller-scale environments or scenarios where computational resources are limited to a single workstation, it falls short for handling massive datasets requiring distributed computing. For this reason, I decided not to pursue this approach further.
That said, this method remains a viable option for clients or use cases where multi-node clusters or distributed systems are not feasible. In such scenarios, chunking combined with efficient in-memory processing could provide a practical solution for implementing transformations within constrained environments.

## My Implementation Explanation

### Codebase Overview
The implementation consists of the following components:

1. **`DataLoaderTransformer`**:
   - A core utility designed to handle large datasets by leveraging Dask for distributed processing. It supports loading data in chunks, applying transformations, and saving the results into multiple output formats (CSV and Parquet).
   - Features include:
     - Configurable parameters for data partitioning and memory management.
     - Efficient data transformations leveraging Dask’s distributed computation model.
     - Compatibility with various file formats, enabling scalability.

2. **`ScalableVGMMNormalizer`**:
   - Implements the Bayesian Gaussian Mixture Model (BGMM) using sklearn for transforming continuous data. It handles the scalability challenge by training the BGMM on a random subset of the data and applying transformations on the full dataset in a chunked manner.
   - Key methods:
     - `fit`: Fits the BGMM model to a sampled subset of data.
     - `transform_chunk`: Applies transformations to a data chunk.
     - `inverse_transform_chunk`: Performs the inverse transformation for evaluation.

3. **Utility Scripts (`main.py`)**:
   - The `main.py` script orchestrates the overall pipeline by configuring and executing the data loader and transformation processes.

4. **Testing (`run_tests.py`, `unit_test.py`, `sanity_check.py`)**:
   - A robust test suite validates the implementation with unit tests for individual components and integration tests for end-to-end functionality.
   - Features:
     - Tests for initialization and correctness of data loader and VGMM normalizer.
     - A sanity check for combined transformation and inverse transformation, ensuring consistency between original and inverse-transformed data.

### Implementation Details and Approach
The implementation focuses on addressing the scalability issues outlined in the task overview:

#### **Data Loading and Transformation**
- The `DataLoaderTransformer` class utilizes Dask to process datasets too large to fit into memory. It partitions the data into chunks for efficient loading, transformation, and saving. This modular approach ensures flexibility and scalability.

#### **Model Training on Subset**
- The `ScalableVGMMNormalizer` fits a BGMM model on a representative subset of the data. This approach mitigates memory constraints while ensuring that the model learns meaningful patterns from the data. Significant components are identified based on weights, and transformations are applied selectively to active components.

#### **Chunk-Based Transformation**
- Data transformations and inverse transformations are applied in chunks using the `transform_chunk` and `inverse_transform_chunk` methods of `ScalableVGMMNormalizer`. This ensures that the system can handle datasets with up to 1 billion rows without overwhelming memory resources.

#### **Evaluation and Metrics**
- After transformations, the `utils.calculate_metrics` function calculates evaluation metrics such as Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE) to validate the accuracy of inverse transformations. These metrics ensure fidelity between the original and inverse-transformed data.

#### **Scalability with Dask**
- Dask was chosen for its native compatibility with Python libraries and its ability to scale computations across multiple nodes and cores. By integrating Dask into the data pipeline, the implementation effectively addresses the scalability challenges posed by massive datasets.

#### **Testing and Validation**
- A comprehensive suite of tests validates individual components (e.g., VGMM normalizer, data loader) and ensures that the combined transformation pipeline produces accurate and reliable results. The test cases also include cleanup procedures to handle temporary files generated during testing.


## Future Improvements

While the current implementation addresses some of the challenges associated with scaling data transformations for massive datasets, there are several areas for improvement and future exploration:

### 1. Scaling the Training of VGMM
- One of the significant limitations of the current implementation is the reliance on a small subset of the original dataset for training the Bayesian Gaussian Mixture Model (BGMM). This approach, while practical in the short term, sacrifices the ability to capture rare patterns or outliers that may be critical for effective modeling.
- The root cause of this limitation is the lack of support for minibatch training in sklearn’s BGMM implementation. This prevents scaling the model training to handle massive datasets directly.
- A promising solution lies in exploring TensorFlow Probability and stochastic variational inference techniques, as discussed in the referenced blog post. These methods allow for minibatch-based training of probabilistic models, making them well-suited for large-scale datasets. If given more time, this would be a key area to focus on for future development.

### 2. Parallelizing Transformation Operations
- The implementation demonstrates how to split sequential tasks, such as transformation operations, into parallel processes using Dask. This approach significantly improves performance by leveraging multi-core and multi-node processing capabilities.
- However, the current implementation only scratches the surface of what is possible with parallel processing. Future iterations of the library should fully explore and optimize parallel transformation pipelines to handle increasingly complex workflows.
- By systematically integrating parallelism into the library, it will be possible to streamline computation-heavy tasks and further enhance scalability.

### 3. Choosing Dask Over Spark
- In the debate between Dask and Spark, I believe Dask is better aligned with the company’s focus on integrating new research ideas into production pipelines. Dask’s compatibility with Python and its support for research-oriented libraries like NumPy, Pandas, and sklearn make it an ideal choice for experimentation and iterative development.
- While Spark excels in large-scale production environments, its reliance on Java-based architecture and limited Python-native capabilities add friction to transitioning from research to production. Dask provides a smoother pathway for incorporating cutting-edge research directly into the pipeline.
- Future work should focus on solidifying the Dask-based infrastructure and expanding its use across different components of the library, ensuring seamless integration of research innovations into scalable production workflows.

---

By addressing these areas, the implementation can be transformed into a robust, scalable system that seamlessly bridges the gap between research and production while efficiently handling datasets at scale.

## References 
1. Link to the blog post: https://brendanhasz.github.io/2019/06/12/tfp-gmm.html