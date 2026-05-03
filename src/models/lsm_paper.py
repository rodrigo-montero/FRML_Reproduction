import numpy as np


class PaperLSM:
    """
    Paper-aligned Liquid State Machine.

    Paper-specified:
    - 3D reservoir grid
    - half excitatory, half inhibitory neurons
    - recurrent probability:
        P(i,j) = C * exp(-(D(i,j) / lambda)^2)
    - C values:
        EE = 0.2, EI = 0.1, IE = 0.05, II = 0.3
    - w_lsm = 1
    - theta = 20
    - LIF dynamics with synaptic current u and voltage v
    - only readout/classifier trained externally

    Assumed:
    - lambda_param
    - input_density
    - input_weight
    - random seed
    """

    def __init__(
        self,
        input_size,
        reservoir_size=4000,
        grid_shape=(20, 20, 10),
        input_density=0.02,
        input_weight=1.0,
        reservoir_weight=1.0,
        lambda_param=3.0,
        threshold=20.0,
        tau_v=5.0,
        tau_u=10.0,
        seed=42,
    ):
        assert np.prod(grid_shape) == reservoir_size, (
            f"grid_shape {grid_shape} must multiply to reservoir_size {reservoir_size}"
        )

        self.input_size = input_size
        self.reservoir_size = reservoir_size
        self.grid_shape = grid_shape

        self.input_density = input_density
        self.input_weight = input_weight
        self.reservoir_weight = reservoir_weight
        self.lambda_param = lambda_param

        self.threshold = threshold
        self.tau_v = tau_v
        self.tau_u = tau_u

        self.rng = np.random.default_rng(seed)

        self.w_in = self._make_standard_input_weights()
        self.w_rec = self._make_distance_based_reservoir_weights()

    def _reservoir_coordinates(self):
        nx, ny, nz = self.grid_shape

        coords = []
        for x in range(nx):
            for y in range(ny):
                for z in range(nz):
                    coords.append((x, y, z))

        return np.array(coords, dtype=np.float32)

    def _neuron_types(self):
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

    def _make_standard_input_weights(self):
        """
        Standard input:
        flattened input connects randomly to reservoir neurons,
        with equal positive and negative signs.
        """
        mask = self.rng.random(
            (self.input_size, self.reservoir_size)
        ) < self.input_density

        signs = self.rng.choice(
            [-1.0, 1.0],
            size=(self.input_size, self.reservoir_size)
        )

        return (mask * signs * self.input_weight).astype(np.float32)

    def _make_distance_based_reservoir_weights(self):
        coords = self._reservoir_coordinates()

        diff = coords[:, None, :] - coords[None, :, :]
        distances = np.linalg.norm(diff, axis=-1)

        C = self._connection_C_matrix()

        probabilities = C * np.exp(
            -((distances / self.lambda_param) ** 2)
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

    def transform_one(self, x):
        """
        x shape: (time_bins, input_size)
        returns: spike counts, shape (reservoir_size,)
        """
        v = np.zeros(self.reservoir_size, dtype=np.float32)
        u = np.zeros(self.reservoir_size, dtype=np.float32)
        spikes = np.zeros(self.reservoir_size, dtype=np.float32)
        spike_counts = np.zeros(self.reservoir_size, dtype=np.float32)

        voltage_decay = 1.0 - (1.0 / self.tau_v)
        current_decay = 1.0 - (1.0 / self.tau_u)

        for t in range(x.shape[0]):
            input_current = x[t] @ self.w_in
            recurrent_current = spikes @ self.w_rec

            u = current_decay * u + input_current + recurrent_current
            v = voltage_decay * v + u

            spikes = (v >= self.threshold).astype(np.float32)
            v[spikes > 0] = 0.0

            spike_counts += spikes

        return spike_counts

    def transform(self, X):
        features = []

        for i, x in enumerate(X):
            features.append(self.transform_one(x))

            if (i + 1) % 25 == 0:
                print(f"Transformed {i + 1}/{len(X)} samples")

        return np.stack(features)


class ReceptiveFieldLSM(PaperLSM):
    """
    Paper-aligned LSM with receptive-field input connections.

    Same reservoir as PaperLSM, but input connections preserve spatial structure:
    each input pixel only connects to reservoir neurons near the corresponding
    (x, y) location in the reservoir grid.
    """

    def __init__(
        self,
        input_shape=(64, 64, 2),
        reservoir_size=4000,
        grid_shape=(20, 20, 10),
        receptive_field_size=6,
        input_density=0.02,
        input_weight=1.0,
        reservoir_weight=1.0,
        lambda_param=3.0,
        threshold=20.0,
        tau_v=5.0,
        tau_u=10.0,
        seed=42,
    ):
        self.input_shape = input_shape
        self.receptive_field_size = receptive_field_size

        input_size = int(np.prod(input_shape))

        super().__init__(
            input_size=input_size,
            reservoir_size=reservoir_size,
            grid_shape=grid_shape,
            input_density=input_density,
            input_weight=input_weight,
            reservoir_weight=reservoir_weight,
            lambda_param=lambda_param,
            threshold=threshold,
            tau_v=tau_v,
            tau_u=tau_u,
            seed=seed,
        )

    def _input_index(self, x, y, p):
        height, width, polarities = self.input_shape
        return (y * width * polarities) + (x * polarities) + p

    def _make_standard_input_weights(self):
        """
        Override PaperLSM standard input with receptive-field input.

        Each input pixel/polarity connects only to reservoir neurons whose
        reservoir (x, y) coordinates are close to the mapped input (x, y).
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