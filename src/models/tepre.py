import numpy as np


class TEPRE:
    """
    Temporal Excitation Partitioned Reservoir Ensemble.

    Paper-aligned implementation based on available information.

    Paper-specified:
    - Reservoir split into temporal partitions
    - Each partition receives input only during its assigned time window
    - Sparse inhibitory connections between partition reservoirs
    - LIF neurons with synaptic current u and membrane voltage v
    - theta = 20
    - dt = 1
    - tau_u = tau_v = 16 for N-MNIST
    - w_lsm = 1
    - 3D reservoir grid
    - distance-based recurrent probability:
        P(i,j) = C * exp(-(D(i,j) / lambda)^2)
    - C values:
        EE = 0.2, EI = 0.1, IE = 0.05, II = 0.3
    - standard input only for TEPRE

    Assumed / not fully specified:
    - input_density
    - input_weight
    - lambda_param
    - inter_partition_density
    - inter_partition connection direction
    - random seed
    """

    def __init__(
        self,
        input_size,
        total_reservoir_size=3600,
        n_partitions=3,
        grid_shape=(10, 10, 12),
        input_density=0.02,
        input_weight=1.0,
        reservoir_weight=1.0,
        inter_partition_density=0.001,
        inter_partition_weight=-1.0,
        lambda_param=3.0,
        threshold=20.0,
        tau_v=16.0,
        tau_u=16.0,
        seed=42,
    ):
        assert total_reservoir_size % n_partitions == 0, (
            "total_reservoir_size must be divisible by n_partitions"
        )

        self.input_size = input_size
        self.total_reservoir_size = total_reservoir_size
        self.n_partitions = n_partitions
        self.partition_size = total_reservoir_size // n_partitions

        assert np.prod(grid_shape) == self.partition_size, (
            f"grid_shape {grid_shape} must multiply to partition_size "
            f"{self.partition_size}"
        )

        self.grid_shape = grid_shape
        self.threshold = threshold
        self.tau_v = tau_v
        self.tau_u = tau_u
        self.lambda_param = lambda_param

        self.rng = np.random.default_rng(seed)

        self.w_in = self._make_input_weights(
            density=input_density,
            weight=input_weight,
        )

        self.w_rec = self._make_distance_based_reservoir_weights(
            weight=reservoir_weight,
        )

        self.w_inter = self._make_inter_partition_weights(
            density=inter_partition_density,
            weight=inter_partition_weight,
        )

        self.w_total = (self.w_rec + self.w_inter).astype(np.float32)

    def _make_input_weights(self, density, weight):
        """
        Standard input connections.

        Each input neuron connects randomly to reservoir neurons.
        We use equal positive and negative signs, following the paper's
        standard input description.
        """
        mask = self.rng.random(
            (self.input_size, self.total_reservoir_size)
        ) < density

        signs = self.rng.choice(
            [-1.0, 1.0],
            size=(self.input_size, self.total_reservoir_size),
        )

        return (mask * signs * weight).astype(np.float32)

    def _partition_coordinates(self):
        """
        Create 3D coordinates for neurons inside one partition reservoir.
        """
        nx, ny, nz = self.grid_shape

        coords = []
        for x in range(nx):
            for y in range(ny):
                for z in range(nz):
                    coords.append((x, y, z))

        return np.array(coords, dtype=np.float32)

    def _neuron_types(self):
        """
        First half excitatory, second half inhibitory.
        Returns:
            True  = excitatory
            False = inhibitory
        """
        is_excitatory = np.ones(self.partition_size, dtype=bool)
        is_excitatory[self.partition_size // 2:] = False
        return is_excitatory

    def _connection_C_matrix(self):
        """
        Build C(i,j) matrix using the paper's constants.

        Source neuron type controls sign later.
        C depends on source and target type.

        EE = source excitatory, target excitatory
        EI = source excitatory, target inhibitory
        IE = source inhibitory, target excitatory
        II = source inhibitory, target inhibitory
        """
        is_exc = self._neuron_types()

        source_exc = is_exc[:, None]
        target_exc = is_exc[None, :]

        C = np.zeros((self.partition_size, self.partition_size), dtype=np.float32)

        C[source_exc & target_exc] = 0.2
        C[source_exc & ~target_exc] = 0.1
        C[~source_exc & target_exc] = 0.05
        C[~source_exc & ~target_exc] = 0.3

        return C

    def _make_single_partition_reservoir(self, weight):
        """
        Distance-based recurrent connectivity for one partition reservoir.
        """
        coords = self._partition_coordinates()

        diff = coords[:, None, :] - coords[None, :, :]
        distances = np.linalg.norm(diff, axis=-1)

        C = self._connection_C_matrix()

        probabilities = C * np.exp(
            -((distances / self.lambda_param) ** 2)
        )

        np.fill_diagonal(probabilities, 0.0)

        mask = self.rng.random(
            (self.partition_size, self.partition_size)
        ) < probabilities

        is_exc = self._neuron_types()

        signs = np.ones((self.partition_size, self.partition_size), dtype=np.float32)
        signs[~is_exc, :] = -1.0

        W = mask * signs * weight
        np.fill_diagonal(W, 0.0)

        return W.astype(np.float32)

    def _make_distance_based_reservoir_weights(self, weight):
        """
        Build block-diagonal recurrent matrix.
        Each temporal partition has its own distance-based reservoir.
        """
        W = np.zeros(
            (self.total_reservoir_size, self.total_reservoir_size),
            dtype=np.float32,
        )

        for p in range(self.n_partitions):
            start = p * self.partition_size
            end = start + self.partition_size

            W[start:end, start:end] = self._make_single_partition_reservoir(
                weight=weight
            )

        return W

    def _make_inter_partition_weights(self, density, weight):
        """
        Sparse inhibitory connections between successive partition reservoirs.

        The paper says these are very sparse inhibitory connections.
        Exact sparsity and direction are not specified.

        Assumption:
        - feed-forward inhibitory connections from partition p to p+1.
        """
        W = np.zeros(
            (self.total_reservoir_size, self.total_reservoir_size),
            dtype=np.float32,
        )

        for p in range(self.n_partitions - 1):
            src_start = p * self.partition_size
            src_end = src_start + self.partition_size

            tgt_start = (p + 1) * self.partition_size
            tgt_end = tgt_start + self.partition_size

            mask = self.rng.random(
                (self.partition_size, self.partition_size)
            ) < density

            W[src_start:src_end, tgt_start:tgt_end] = mask * weight

        return W.astype(np.float32)

    def _partition_mask_for_time(self, t, n_time_bins):
        """
        Only one partition receives external input at timestep t.
        """
        partition_id = min(
            int(t / n_time_bins * self.n_partitions),
            self.n_partitions - 1,
        )

        mask = np.zeros(self.total_reservoir_size, dtype=np.float32)

        start = partition_id * self.partition_size
        end = start + self.partition_size

        mask[start:end] = 1.0
        return mask

    def transform_one(self, x):
        """
        x shape: (time_bins, input_size)

        Returns:
            spike-count feature vector of shape (total_reservoir_size,)
        """
        n_time_bins = x.shape[0]

        v = np.zeros(self.total_reservoir_size, dtype=np.float32)
        u = np.zeros(self.total_reservoir_size, dtype=np.float32)
        spikes = np.zeros(self.total_reservoir_size, dtype=np.float32)
        spike_counts = np.zeros(self.total_reservoir_size, dtype=np.float32)

        voltage_decay = 1.0 - (1.0 / self.tau_v)
        current_decay = 1.0 - (1.0 / self.tau_u)

        for t in range(n_time_bins):
            partition_mask = self._partition_mask_for_time(t, n_time_bins)

            input_current = (x[t] @ self.w_in) * partition_mask
            recurrent_current = spikes @ self.w_total

            u = current_decay * u + input_current + recurrent_current
            v = voltage_decay * v + u

            spikes = (v >= self.threshold).astype(np.float32)

            # Reset membrane voltage after spike.
            v[spikes > 0] = 0.0

            spike_counts += spikes

        return spike_counts

    def transform(self, X):
        """
        X shape: (n_samples, time_bins, input_size)

        Returns:
            feature matrix of shape (n_samples, total_reservoir_size)
        """
        features = []

        for i, x in enumerate(X):
            features.append(self.transform_one(x))

            if (i + 1) % 100 == 0:
                print(f"Transformed {i + 1}/{len(X)} samples")

        return np.stack(features)