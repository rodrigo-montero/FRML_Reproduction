import numpy as np


class SimpleLSM:
    def __init__(
        self,
        input_size,
        reservoir_size=500,
        input_density=0.05,
        reservoir_density=0.02,
        input_weight=1.0,
        reservoir_weight=1.0,
        threshold=20.0,
        tau_v=16.0,
        seed=42,
    ):
        self.input_size = input_size
        self.reservoir_size = reservoir_size
        self.threshold = threshold
        self.tau_v = tau_v
        self.rng = np.random.default_rng(seed)

        self.w_in = self._make_input_weights(input_density, input_weight)
        self.w_rec = self._make_reservoir_weights(reservoir_density, reservoir_weight)

    def _make_input_weights(self, density, weight):
        mask = self.rng.random((self.input_size, self.reservoir_size)) < density
        signs = self.rng.choice([-1.0, 1.0], size=(self.input_size, self.reservoir_size))
        return (mask * signs * weight).astype(np.float32)

    def _make_reservoir_weights(self, density, weight):
        mask = self.rng.random((self.reservoir_size, self.reservoir_size)) < density
        np.fill_diagonal(mask, 0)

        signs = np.ones((self.reservoir_size, self.reservoir_size), dtype=np.float32)
        inhibitory = np.arange(self.reservoir_size) >= self.reservoir_size // 2
        signs[inhibitory, :] = -1.0

        return (mask * signs * weight).astype(np.float32)

    def transform_one(self, x):
        """
        x shape: (time_bins, input_size)
        returns reservoir spike-count features: (reservoir_size,)
        """
        v = np.zeros(self.reservoir_size, dtype=np.float32)
        spikes = np.zeros(self.reservoir_size, dtype=np.float32)
        spike_counts = np.zeros(self.reservoir_size, dtype=np.float32)

        decay = 1.0 - (1.0 / self.tau_v)

        for t in range(x.shape[0]):
            input_current = x[t] @ self.w_in
            recurrent_current = spikes @ self.w_rec

            v = decay * v + input_current + recurrent_current

            spikes = (v >= self.threshold).astype(np.float32)
            v[spikes > 0] = 0.0

            spike_counts += spikes

        return spike_counts

    def transform(self, X):
        """
        X shape: (n_samples, time_bins, input_size)
        returns: (n_samples, reservoir_size)
        """
        features = []
        for i, x in enumerate(X):
            features.append(self.transform_one(x))
            if (i + 1) % 100 == 0:
                print(f"Transformed {i + 1}/{len(X)} samples")

        return np.stack(features)