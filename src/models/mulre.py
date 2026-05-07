import numpy as np
from scipy.signal import fftconvolve
from scipy.sparse import csr_matrix


def build_gabor_filter_bank(n_filters=18, ksize=9, sigma=2.0, lam=4.0, gamma=0.5):
    """
    Build a bank of 18 Gabor filters evenly spaced in orientation [0, pi).

    The paper references Gabor filters as texture discriminators (citation [14]:
    Fogel & Sagi, 1989). A standard parameterisation is used here; the exact
    sigma/lambda/gamma values are not given in the paper.

    Args:
        n_filters:  number of orientations (paper specifies 18)
        ksize:      kernel spatial size (ksize x ksize)
        sigma:      Gaussian envelope standard deviation
        lam:        sinusoid wavelength
        gamma:      spatial aspect ratio

    Returns:
        kernel_stack: (n_filters, ksize, ksize) float32 array
    """
    thetas = np.linspace(0, np.pi, n_filters, endpoint=False)

    half = ksize // 2
    x, y = np.meshgrid(np.arange(-half, half + 1), np.arange(-half, half + 1))
    # shape: (ksize, ksize)

    cos_t = np.cos(thetas)[:, None, None]   # (F, 1, 1)
    sin_t = np.sin(thetas)[:, None, None]

    x_rot =  x[None] * cos_t + y[None] * sin_t   # (F, ksize, ksize)
    y_rot = -x[None] * sin_t + y[None] * cos_t

    envelope = np.exp(-(x_rot**2 + gamma**2 * y_rot**2) / (2 * sigma**2))
    carrier  = np.cos(2 * np.pi * x_rot / lam)

    kernels = (envelope * carrier).astype(np.float32)           # (F, ksize, ksize)
    kernels -= kernels.mean(axis=(1, 2), keepdims=True)         # zero-mean each

    return kernels   # (n_filters, ksize, ksize)


def apply_gabor_bank_to_sequence(x, sensor_shape=(34, 34, 2), kernels=None):
    """
    Apply Gabor filter bank to an N-MNIST frame sequence.

    Uses batched FFT convolution (numpy.fft.rfft2) over all frames and filters
    simultaneously — one FFT per filter over the entire (T*polarities, H, W)
    batch, which is significantly faster than per-frame convolution.

    Args:
        x:            (T, H*W*polarities) flattened frame sequence
        sensor_shape: (H, W, polarities) — (34, 34, 2) for N-MNIST
        kernels:      (n_filters, ksize, ksize) float32, from build_gabor_filter_bank

    Returns:
        x_out: (T, H * W * polarities * n_filters) float32
    """
    from numpy.fft import rfft2, irfft2

    H, W, polarities = sensor_shape
    T = x.shape[0]
    n_filters = kernels.shape[0]

    # (T*polarities, H, W)
    x_batch = x.reshape(T, H, W, polarities).transpose(0, 3, 1, 2).reshape(-1, H, W)
    B = x_batch.shape[0]

    # Pre-compute FFT of all input frames once: (B, H, W//2+1)
    X_f = rfft2(x_batch, s=(H, W))

    responses = np.empty((n_filters, B, H, W), dtype=np.float32)

    for f in range(n_filters):
        # Zero-pad kernel to (H, W) and FFT
        k_pad = np.zeros((H, W), dtype=np.float32)
        kh, kw = kernels[f].shape
        k_pad[:kh, :kw] = kernels[f]
        K_f = rfft2(k_pad, s=(H, W))               # (H, W//2+1)

        resp = irfft2(X_f * K_f[None], s=(H, W)).astype(np.float32)  # (B, H, W)
        np.maximum(resp, 0.0, out=resp)            # half-wave rectify in-place
        responses[f] = resp

    # (n_filters, B, H, W) -> (B, H, W, n_filters) -> (T, pol, H, W, F) -> (T, H, W, pol*F)
    responses = responses.transpose(1, 2, 3, 0)                   # (B, H, W, F)
    responses = responses.reshape(T, polarities, H, W, n_filters)
    responses = responses.transpose(0, 2, 3, 1, 4)                # (T, H, W, pol, F)

    return responses.reshape(T, -1).astype(np.float32)


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
    - N-MNIST frames preprocessed by a bank of 18 Gabor filters

    Assumed / not fully specified:
    - lambda_param
    - input_density
    - input_weight
    - receptive-field window size
    - Gabor filter hyperparameters (sigma, lambda, gamma, ksize)
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
        # Gabor filter bank
        use_gabor=True,
        n_gabor_filters=18,
        gabor_ksize=9,
        gabor_sigma=2.0,
        gabor_lam=4.0,
        gabor_gamma=0.5,
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
        self.use_gabor = use_gabor

        if use_gabor:
            self.gabor_kernels = build_gabor_filter_bank(
                n_filters=n_gabor_filters,
                ksize=gabor_ksize,
                sigma=gabor_sigma,
                lam=gabor_lam,
                gamma=gabor_gamma,
            )
            H, W, polarities = input_shape
            self.gabor_input_shape = (H, W, polarities * n_gabor_filters)
            self.input_size = int(np.prod(self.gabor_input_shape))
        else:
            self.gabor_kernels = None
            self.gabor_input_shape = input_shape
            self.input_size = int(np.prod(input_shape))

        self.total_reservoir_size = total_reservoir_size
        self.n_reservoirs = n_reservoirs
        self.reservoir_size = total_reservoir_size // n_reservoirs

        assert np.prod(grid_shape) == self.reservoir_size, (
            f"grid_shape {grid_shape} must multiply to reservoir_size {self.reservoir_size}"
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
            self.w_in_list.append(self._make_receptive_field_input_weights())
            self.w_rec_list.append(self._make_distance_biased_reservoir_weights(d=d))

    # ------------------------------------------------------------------
    # Reservoir geometry helpers
    # ------------------------------------------------------------------

    def _reservoir_coordinates(self):
        nx, ny, nz = self.grid_shape
        xs, ys, zs = np.meshgrid(
            np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij"
        )
        return np.stack([xs.ravel(), ys.ravel(), zs.ravel()], axis=1).astype(np.float32)

    def _neuron_types(self):
        is_exc = np.ones(self.reservoir_size, dtype=bool)
        is_exc[self.reservoir_size // 2:] = False
        return is_exc

    def _connection_C_matrix(self):
        is_exc = self._neuron_types()
        src_exc = is_exc[:, None]
        tgt_exc = is_exc[None, :]
        C = np.zeros((self.reservoir_size, self.reservoir_size), dtype=np.float32)
        C[ src_exc &  tgt_exc] = 0.2   # EE
        C[ src_exc & ~tgt_exc] = 0.1   # EI
        C[~src_exc &  tgt_exc] = 0.05  # IE
        C[~src_exc & ~tgt_exc] = 0.3   # II
        return C

    # ------------------------------------------------------------------
    # Weight constructors
    # ------------------------------------------------------------------

    def _make_distance_biased_reservoir_weights(self, d):
        """P(i,j) = C * exp(-((D(i,j) - d) / lambda)^2)  [Eq. 5 in paper]"""
        coords = self._reservoir_coordinates()
        diff = coords[:, None, :] - coords[None, :, :]
        distances = np.linalg.norm(diff, axis=-1)

        C = self._connection_C_matrix()
        probs = C * np.exp(-(((distances - d) / self.lambda_param) ** 2))
        np.fill_diagonal(probs, 0.0)

        mask = self.rng.random((self.reservoir_size, self.reservoir_size)) < probs

        is_exc = self._neuron_types()
        signs = np.ones((self.reservoir_size, self.reservoir_size), dtype=np.float32)
        signs[~is_exc, :] = -1.0

        W = mask * signs * self.reservoir_weight
        np.fill_diagonal(W, 0.0)
        return W.astype(np.float32)

    def _make_receptive_field_input_weights(self):
        """
        Vectorized receptive-field input weight construction.

        Instead of looping over every (x, y, channel) input neuron, we:
        1. Build a lookup table mapping each reservoir neuron to its (mapped_x, mapped_y).
        2. For each spatial position in the input grid, find all candidate reservoir
           neurons in one NumPy operation.
        3. Assign connections for all channels at that position simultaneously.
        """
        H, W, n_channels = self.gabor_input_shape
        nx, ny, _ = self.grid_shape

        reservoir_coords = self._reservoir_coordinates()
        res_x = reservoir_coords[:, 0].astype(np.int32)
        res_y = reservoir_coords[:, 1].astype(np.int32)

        half_w = self.receptive_field_size // 2

        # Pre-map every input (x, y) to its reservoir (x, y)
        ix_all = np.arange(W)
        iy_all = np.arange(H)
        mapped_ix = np.round((ix_all / max(W - 1, 1)) * (nx - 1)).astype(np.int32)
        mapped_iy = np.round((iy_all / max(H - 1, 1)) * (ny - 1)).astype(np.int32)

        # Collect COO entries for sparse construction then densify
        rows_list = []
        cols_list = []
        vals_list = []

        for iy in range(H):
            for ix in range(W):
                mx, my = mapped_ix[ix], mapped_iy[iy]

                candidates = np.where(
                    (np.abs(res_x - mx) <= half_w) &
                    (np.abs(res_y - my) <= half_w)
                )[0]

                if len(candidates) == 0:
                    continue

                # For all n_channels at this (x, y): draw connections in one shot
                # Shape of random draw: (n_channels, len(candidates))
                rand = self.rng.random((n_channels, len(candidates)))
                connected = rand < self.input_density   # bool mask

                ch_idx, cand_idx = np.where(connected)
                if len(ch_idx) == 0:
                    continue

                # input flat index: (iy * W + ix) * n_channels + ch
                input_rows = (iy * W + ix) * n_channels + ch_idx
                reservoir_cols = candidates[cand_idx]

                signs = self.rng.choice(
                    np.array([-1.0, 1.0], dtype=np.float32), size=len(ch_idx)
                )

                rows_list.append(input_rows)
                cols_list.append(reservoir_cols)
                vals_list.append(signs * self.input_weight)

        W_mat = np.zeros((self.input_size, self.reservoir_size), dtype=np.float32)
        if rows_list:
            rows = np.concatenate(rows_list)
            cols = np.concatenate(cols_list)
            vals = np.concatenate(vals_list)
            W_mat[rows, cols] = vals

        return csr_matrix(W_mat)  # store sparse — ~2% density makes matmul ~50x faster

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------

    def _preprocess(self, x):
        """Apply Gabor bank if enabled. x: (T, H*W*pol) -> (T, H*W*pol*n_filt)"""
        if not self.use_gabor:
            return x
        return apply_gabor_bank_to_sequence(
            x,
            sensor_shape=self.input_shape,
            kernels=self.gabor_kernels,
        )

    def _run_single_reservoir(self, x, w_in, w_rec):
        """x: (T, input_size) -> spike_counts: (reservoir_size,)"""
        v = np.zeros(self.reservoir_size, dtype=np.float32)
        u = np.zeros(self.reservoir_size, dtype=np.float32)
        spikes = np.zeros(self.reservoir_size, dtype=np.float32)
        spike_counts = np.zeros(self.reservoir_size, dtype=np.float32)

        v_decay = 1.0 - 1.0 / self.tau_v
        u_decay = 1.0 - 1.0 / self.tau_u

        # Pre-compute all input projections at once: (T, reservoir_size)
        # w_in is stored as a CSR sparse matrix (~2% density).
        # Sparse matmul is ~50x faster than dense for this sparsity level.
        input_projections = (x @ w_in).toarray() if hasattr(x @ w_in, 'toarray') else x @ w_in
        # Cleaner: use the sparse matrix directly
        input_projections = np.asarray(w_in.T.dot(x.T).T, dtype=np.float32)  # (T, res)

        for t in range(x.shape[0]):
            u = u_decay * u + input_projections[t] + spikes @ w_rec
            v = v_decay * v + u

            spikes = (v >= self.threshold).astype(np.float32)
            v[spikes > 0] = 0.0
            spike_counts += spikes

        return spike_counts

    def transform_one(self, x):
        """x: (T, H*W*pol) -> (total_reservoir_size,)"""
        x_proc = self._preprocess(x)
        return np.concatenate([
            self._run_single_reservoir(x_proc, w_in, w_rec)
            for w_in, w_rec in zip(self.w_in_list, self.w_rec_list)
        ])

    def transform(self, X):
        """X: (N, T, H*W*pol) -> (N, total_reservoir_size)"""
        # Preprocess the entire dataset with Gabor in one batched FFT call.
        # We treat (N*T*polarities) as one big frame batch, compute FFTs once,
        # then filter with each of the 18 kernels in a single rfft2 multiply.
        if self.use_gabor:
            print("  Applying Gabor filter bank to dataset...", flush=True)
            from numpy.fft import rfft2, irfft2

            H, W, polarities = self.input_shape
            N, T, _ = X.shape
            n_filters = self.gabor_kernels.shape[0]

            # (N*T*pol, H, W)
            x_all = X.reshape(N * T, H, W, polarities).transpose(0, 3, 1, 2).reshape(-1, H, W)
            B = x_all.shape[0]

            # FFT of every frame once
            X_f = rfft2(x_all, s=(H, W))   # (B, H, W//2+1)

            responses = np.empty((n_filters, B, H, W), dtype=np.float32)
            for f in range(n_filters):
                k_pad = np.zeros((H, W), dtype=np.float32)
                kh, kw = self.gabor_kernels[f].shape
                k_pad[:kh, :kw] = self.gabor_kernels[f]
                K_f = rfft2(k_pad, s=(H, W))
                resp = irfft2(X_f * K_f[None], s=(H, W)).astype(np.float32)
                np.maximum(resp, 0.0, out=resp)
                responses[f] = resp

            # -> (N, T, H, W, pol, F) -> (N, T, H*W*pol*F)
            responses = responses.transpose(1, 2, 3, 0)                  # (B, H, W, F)
            responses = responses.reshape(N, T, polarities, H, W, n_filters)
            responses = responses.transpose(0, 1, 3, 4, 2, 5)            # (N, T, H, W, pol, F)
            X_proc = responses.reshape(N, T, -1).astype(np.float32)
            print("  Gabor done.", flush=True)
        else:
            X_proc = X

        features = []
        for i, x in enumerate(X_proc):
            features.append(np.concatenate([
                self._run_single_reservoir(x, w_in, w_rec)
                for w_in, w_rec in zip(self.w_in_list, self.w_rec_list)
            ]))
            if (i + 1) % 100 == 0:
                print(f"  Transformed {i + 1}/{len(X_proc)} samples", flush=True)
        return np.stack(features)