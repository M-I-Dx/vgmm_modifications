import numpy as np
from sklearn.mixture import BayesianGaussianMixture


class ScalableVGMMNormalizer:
    def __init__(self, config=None):
        """
        Scalable VGMM Normalizer to handle large datasets by fitting on a subset
        and applying transformations on the full dataset.

        Parameters:
            config (dict): Configuration dictionary for model parameters.
        """
        # Default configuration
        default_config = {
            "n_clusters": 10,
            "eps": 0.005,
            "weight_concentration_prior_type": "dirichlet_process",
            "weight_concentration_prior": 0.001,
            "max_iter": 100,
            "n_init": 1,
            "random_state": 42,
        }

        # Update default configuration with user-provided values
        self.config = {**default_config, **(config or {})}

        self.n_clusters = self.config["n_clusters"]
        self.eps = self.config["eps"]
        self.model = None
        self.components = []
        self.ordering = None

    def fit(self, sample_data: np.ndarray):
        """
        Fit the VGMM model on a sampled subset of the data.

        Parameters:
            sample_data (np.ndarray): Sampled data to fit the model.
        """
        self.model = BayesianGaussianMixture(
            n_components=self.n_clusters,
            weight_concentration_prior_type=self.config[
                "weight_concentration_prior_type"
            ],
            weight_concentration_prior=self.config["weight_concentration_prior"],
            max_iter=self.config["max_iter"],
            n_init=self.config["n_init"],
            random_state=self.config["random_state"],
        )
        self.model.fit(sample_data.reshape(-1, 1))
        significant_components = self.model.weights_ > self.eps
        mode_freq = np.unique(self.model.predict(sample_data.reshape(-1, 1)))
        self.components = [
            comp if (idx in mode_freq and significant_components[idx]) else False
            for idx, comp in enumerate(significant_components)
        ]
        self.ordering = np.argsort(
            -self.model.weights_[self.components]
        )  # Restrict ordering to active components

    def transform_chunk(self, data_chunk: np.ndarray):
        """
        Transform a chunk of data using the pre-fitted model.

        Parameters:
            data_chunk (np.ndarray): A chunk of data to transform.

        Returns:
            np.ndarray: Transformed data.
        """
        data = data_chunk.reshape(-1, 1)
        means = self.model.means_.reshape((1, self.n_clusters))
        stds = np.sqrt(self.model.covariances_).reshape((1, self.n_clusters))
        features = (data - means) / (4 * stds)
        features = features[:, self.components]
        n_opts = sum(self.components)
        probs = self.model.predict_proba(data)
        probs = probs[:, self.components]
        opt_sel = np.array(
            [np.random.choice(np.arange(n_opts), p=pp / pp.sum()) for pp in probs]
        )
        probs_onehot = np.zeros_like(probs)
        probs_onehot[np.arange(len(probs)), opt_sel] = 1

        idx = np.arange(len(features))
        selected_features = features[idx, opt_sel].reshape(-1, 1)
        selected_features = np.clip(selected_features, -0.99, 0.99)

        reordered_probs_onehot = np.zeros_like(probs_onehot)
        for id, val in enumerate(self.ordering):
            if id < reordered_probs_onehot.shape[1]:
                reordered_probs_onehot[:, id] = probs_onehot[:, val]

        return np.concatenate([selected_features, reordered_probs_onehot], axis=1)

    def inverse_transform_chunk(self, transformed_chunk: np.ndarray):
        """
        Inverse transform a chunk of data using the pre-fitted model.

        Parameters:
            transformed_chunk (np.ndarray): Transformed data to revert.

        Returns:
            np.ndarray: Original data.
        """
        u = np.clip(transformed_chunk[:, 0], -1, 1)
        v = transformed_chunk[:, 1:]

        v_reordered = np.zeros_like(v)
        for idx, val in enumerate(self.ordering[: v.shape[1]]):
            v_reordered[:, val] = v[:, idx]

        v_t = np.ones((len(u), self.n_clusters)) * -100
        v_t[:, self.components] = v_reordered
        p_argmax = np.argmax(v_t, axis=1)

        means = self.model.means_.reshape(-1)
        stds = np.sqrt(self.model.covariances_).reshape(-1)
        mean_t = means[p_argmax]
        std_t = stds[p_argmax]

        original_data = u * 4 * std_t + mean_t
        return original_data.reshape(-1, 1)
