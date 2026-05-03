import numpy as np


class MuLRE:
    """
    Multi-Length Scale Reservoir Ensemble.

    Paper-aligned implementation based on available information.

    Paper-specified:
    - Ensemble of LSM reservoirs
    - Each reservoir uses different distance-bias parameter d
    - For 3 reservoirs: d = {0, 4, 6}
    - Reservoir is a 3D grid
    - Recurrent probability:
        P(i,j) = C * exp(-((D(i,j) - d) / lambda)^2)
    - C values:
        EE = 0.2, EI = 0.1, IE = 0.05, II = 0.3
    - Half neurons excitatory, half inhibitory
    - w_lsm = 1
    - theta = 20
    - tau_u = tau_v = 16 for N-MNIST
    - MuLRE must use receptive-field input connections

    Assumed / not fully specified:
    - lambda_param
    - input_density
    - input_weight
    - receptive-field window size
    - exact Gabor preprocessing omitted in this initial implementation
    - random seed
    """

    def __init__(
        self,
        input_shape=(34, 34, 2),
        total_reservoir_size=3600,
        n_reservoirs=3,
        grid_shape=(10, 10, 12),
        d_values=None,
        receptive_field_size=6,
        input_density=0.02,
        input_weight=1.0,
        reservoir_weight=1.0,
        lambda_param=3.0,
        threshold=20.0,
        tau_v=16.0,
        tau_u=16.0,
        seed=42,
    ):
        if d_values is None:
            if n_reservoirs == 2:
                d_values = [0, 5]
            elif n_reservoirs == 3:
                d_values = [0, 4, 6]
            else:
                d_values = list(range(n_reservoirs))

        assert len(d_values) == n_reservoirs

        assert total_reservoir_size % n_reservoirs == 0, (
            "total_reservoir_size must be divisible by n_reservoirs"
        )

        self.input_shape = input_shape
        self.input_size = int(np.prod(input_shape))

        self.total_reservoir_size = total_reservoir_size
        self.n_reservoirs = n_reservoirs
        self.reservoir_size = total_reservoir_size // n_reservoirs

        assert np.prod(grid_shape) == self.reservoir_size, (
            f"grid_shape {grid_shape} must multiply to reservoir_size "
            f"{self.reservoir_size}"
        )

        self.grid_shape = grid_shape
        self.d_values = d_values
        self.receptive_field_size = receptive_field_size

        self.input_density = input_density
        self.input_weight = input_weight
        self.reservoir_weight = reservoir_weight
        self.lambda_param = lambda_param

        self.threshold = threshold
        self.tau_v = tau_v
        self.tau_u = tau_u

        self.rng = np.random.default_rng(seed)

        self.w_in_list = []
        self.w_rec_list = []

        for r, d in enumerate(self.d_values):
            self.w_in_list.append(
                self._make_receptive_field_input_weights(
                    seed_offset=r
                )
            )

            self.w_rec_list.append(
                self._make_distance_biased_reservoir_weights(
                    d=d
                )
            )

    def _reservoir_coordinates(self):
        nx, ny, nz = self.grid_shape

        coords = []
        for x in range(nx):
            for y in range(ny):
                for z in range(nz):
                    coords.append((x, y, z))

        return np.array(coords, dtype=np.float32)

    def _neuron_types(self):
        """
        True = excitatory, False = inhibitory.
        """
        is_exc = np.ones(self.reservoir_size, dtype=bool)
        is_exc[self.reservoir_size // 2:] = False
        return is_exc

    def _connection_C_matrix(self):
        is_exc = self._neuron_types()

        source_exc = is_exc[:, None]
        target_exc = is_exc[None, :]

        C = np.zeros((self.reservoir_size, self.reservoir_size), dtype=np.float32)

        C[source_exc & target_exc] = 0.2
        C[source_exc & ~target_exc] = 0.1
        C[~source_exc & target_exc] = 0.05
        C[~source_exc & ~target_exc] = 0.3

        return C

    def _make_distance_biased_reservoir_weights(self, d):
        """
        MuLRE recurrent probability:

            P(i,j) = C * exp(-((D(i,j) - d) / lambda)^2)
        """
        coords = self._reservoir_coordinates()

        diff = coords[:, None, :] - coords[None, :, :]
        distances = np.linalg.norm(diff, axis=-1)

        C = self._connection_C_matrix()

        probabilities = C * np.exp(
            -(((distances - d) / self.lambda_param) ** 2)
        )

        np.fill_diagonal(probabilities, 0.0)

        mask = self.rng.random(
            (self.reservoir_size, self.reservoir_size)
        ) < probabilities

        is_exc = self._neuron_types()

        signs = np.ones((self.reservoir_size, self.reservoir_size), dtype=np.float32)
        signs[~is_exc, :] = -1.0

        W = mask * signs * self.reservoir_weight
        np.fill_diagonal(W, 0.0)

        return W.astype(np.float32)

    def _input_index(self, x, y, p):
        height, width, polarities = self.input_shape
        return (y * width * polarities) + (x * polarities) + p

    def _make_receptive_field_input_weights(self, seed_offset=0):
        """
        Receptive-field input connections.

        Each input pixel/polarity connects only to reservoir neurons whose
        reservoir (x,y) coordinates lie inside a local window around the
        corresponding input (x,y) location.

        This preserves spatial order, as described in the paper.
        """
        height, width, polarities = self.input_shape
        nx, ny, nz = self.grid_shape

        W = np.zeros((self.input_size, self.reservoir_size), dtype=np.float32)

        reservoir_coords = self._reservoir_coordinates()

        res_x = reservoir_coords[:, 0]
        res_y = reservoir_coords[:, 1]

        half_window = self.receptive_field_size // 2

        for y in range(height):
            for x in range(width):
                mapped_x = int(round((x / max(width - 1, 1)) * (nx - 1)))
                mapped_y = int(round((y / max(height - 1, 1)) * (ny - 1)))

                candidate_neurons = np.where(
                    (np.abs(res_x - mapped_x) <= half_window) &
                    (np.abs(res_y - mapped_y) <= half_window)
                )[0]

                if len(candidate_neurons) == 0:
                    continue

                for p in range(polarities):
                    input_idx = self._input_index(x, y, p)

                    connect_mask = (
                        self.rng.random(len(candidate_neurons)) < self.input_density
                    )

                    selected = candidate_neurons[connect_mask]

                    if len(selected) == 0:
                        continue

                    signs = self.rng.choice([-1.0, 1.0], size=len(selected))
                    W[input_idx, selected] = signs * self.input_weight

        return W.astype(np.float32)

    def _run_single_reservoir(self, x, w_in, w_rec):
        """
        x shape: (time_bins, input_size)
        returns: spike counts for one reservoir
        """
        v = np.zeros(self.reservoir_size, dtype=np.float32)
        u = np.zeros(self.reservoir_size, dtype=np.float32)
        spikes = np.zeros(self.reservoir_size, dtype=np.float32)
        spike_counts = np.zeros(self.reservoir_size, dtype=np.float32)

        voltage_decay = 1.0 - (1.0 / self.tau_v)
        current_decay = 1.0 - (1.0 / self.tau_u)

        for t in range(x.shape[0]):
            input_current = x[t] @ w_in
            recurrent_current = spikes @ w_rec

            u = current_decay * u + input_current + recurrent_current
            v = voltage_decay * v + u

            spikes = (v >= self.threshold).astype(np.float32)
            v[spikes > 0] = 0.0

            spike_counts += spikes

        return spike_counts

    def transform_one(self, x):
        """
        x shape: (time_bins, input_size)
        output shape: (total_reservoir_size,)
        """
        features = []

        for w_in, w_rec in zip(self.w_in_list, self.w_rec_list):
            z = self._run_single_reservoir(x, w_in, w_rec)
            features.append(z)

        return np.concatenate(features)

    def transform(self, X):
        features = []

        for i, x in enumerate(X):
            features.append(self.transform_one(x))

            if (i + 1) % 100 == 0:
                print(f"Transformed {i + 1}/{len(X)} samples")

        return np.stack(features)