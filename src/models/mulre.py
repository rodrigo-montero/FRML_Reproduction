import numpy as np
from scipy.sparse import csr_matrix

def build_gabor_filter_bank(thetas, lambdas, ksize=5, sigma=10.0, gamma=0.5):
    """
    Build a Gabor filter bank matching the author's implementation.

    The author uses cv2.getGaborKernel with:
        kernel_size = 5, sigma = 10.0, gamma = 0.5, phi = pi/2
    and normalises each kernel by its L2 norm.
    Each (theta, lambda) pair produces one filter that is stacked for both
    polarities: shape (2, ksize, ksize).

    The paper specifies 18 Gabor filters (citation [14]: Fogel & Sagi, 1989).
    The author iterates over thetas × lambdas, so e.g. 9 thetas × 2 lambdas
    = 18 filters total.  We reproduce that scheme here in pure numpy/scipy
    (no OpenCV dependency) while exactly matching the kernel formula.

    Args:
        thetas:  list/array of orientations in degrees
        lambdas: list/array of wavelengths
        ksize:   kernel spatial size (author uses 5)
        sigma:   Gaussian envelope std (author uses 10.0)
        gamma:   spatial aspect ratio (author uses 0.5)

    Returns:
        filters: (n_filters, 2, ksize, ksize) float32 array,
                 where n_filters = len(thetas) * len(lambdas)
    """
    phi = np.pi / 2  # phase offset, as in author's code
    half = ksize // 2
    x, y = np.meshgrid(np.arange(-half, half + 1), np.arange(-half, half + 1))

    filters = []
    for theta_deg in thetas:
        theta = np.radians(theta_deg)
        for lam in lambdas:
            x_rot =  x * np.cos(theta) + y * np.sin(theta)
            y_rot = -x * np.sin(theta) + y * np.cos(theta)

            envelope = np.exp(-(x_rot**2 + gamma**2 * y_rot**2) / (2 * sigma**2))
            carrier  = np.cos(2 * np.pi * x_rot / lam + phi)
            kernel   = (envelope * carrier).astype(np.float64)

            # L2-normalise, matching author's `kernel / np.linalg.norm(kernel)`
            norm = np.linalg.norm(kernel)
            if norm > 0:
                kernel = kernel / norm
            kernel = kernel.astype(np.float32)

            # Stack for both polarities, matching author's torch.stack([g, g], dim=0)
            filters.append(np.stack([kernel, kernel], axis=0))  # (2, ksize, ksize)

    return np.stack(filters, axis=0)  # (n_filters, 2, ksize, ksize)


def apply_gabor_bank(x_frame, kernels):
    """
    Apply Gabor filter bank to a single frame via FFT convolution.

    Args:
        x_frame: (H, W, polarities) float32
        kernels: (n_filters, 2, ksize, ksize) float32  [polarity-paired]

    Returns:
        (H, W, polarities * n_filters) float32
    """
    from numpy.fft import rfft2, irfft2

    H, W, polarities = x_frame.shape
    n_filters = kernels.shape[0]

    # (polarities, H, W)
    x_pol = x_frame.transpose(2, 0, 1).astype(np.float32)
    X_f   = rfft2(x_pol, s=(H, W))  # (polarities, H, W//2+1)

    out = np.empty((n_filters, polarities, H, W), dtype=np.float32)
    for f in range(n_filters):
        for p in range(polarities):
            k_pad = np.zeros((H, W), dtype=np.float32)
            kh, kw = kernels[f, p].shape
            k_pad[:kh, :kw] = kernels[f, p]
            K_f = rfft2(k_pad, s=(H, W))
            resp = np.real(np.fft.irfft2(X_f[p] * K_f, s=(H, W))).astype(np.float32)
            np.maximum(resp, 0.0, out=resp)  # half-wave rectify
            out[f, p] = resp

    # (n_filters, polarities, H, W) -> (H, W, polarities * n_filters)
    out = out.transpose(2, 3, 1, 0)          # (H, W, polarities, n_filters)
    return out.reshape(H, W, polarities * n_filters)


# ---------------------------------------------------------------------------
# MuLRE
# ---------------------------------------------------------------------------

class MuLRE:
    """
    Multi-Length Scale Reservoir Ensemble (MuLRE).

    Paper-specified values are faithfully reproduced.  Values that the paper
    leaves unspecified are taken from the author's reference implementation
    (lsm_weight_definitions.py / lsm_models.py).

    Paper:
        - 3 reservoirs with d = {0, 4, 6} (2-reservoir: {0, 5})
        - P(i,j) = C * exp(-((D(i,j) - d) / lambda)^2)   [Eq. 5]
        - C: EE=0.2, EI=0.1, IE=0.05, II=0.3
        - wlsm = 1  (excitatory +1, inhibitory -1)
        - theta = 20, tau_u = tau_v = 16  (N-MNIST)
        - 3-D reservoir grid, receptive-field input, 18 Gabor filters

    Author's code (unspecified in paper):
        - lambda = 9
        - D is squared Euclidean distance (not √D), used as -D/lambda
          i.e. P = C * exp(-D_sq / lam)  for the *baseline* formula (Eq. 4)
          For MuLRE Eq. 5 the author uses -(sqrt(D_sq) - d)^2 / lam
        - inhibitory fraction = 0.2  (first 20 % of a random permutation)
        - input connections: exactly floor(N * density) positive and the
          same number of negative, chosen from a random shuffle of candidate
          reservoir neurons (receptive-field window)
        - receptive-field mapping: res_x = floor(x * Nx / inW)  (not round)
          window clamped so it never exceeds [0, Nx)
        - curr_prefac = 1 / tau_u multiplied into both W_in and W_lsm
        - Gabor: ksize=5, sigma=10.0, gamma=0.5, phi=pi/2, L2-normalised;
                 thetas (degrees) × lambdas grid → n_filters total
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
        lambda_param=9.0,          # author uses 9
        inh_fraction=0.2,          # author uses 0.2
        threshold=20.0,
        tau_v=16.0,
        tau_u=16.0,
        seed=42,
        # Gabor parameters matching author's implementation
        use_gabor=True,
        gabor_thetas=None,         # degrees; default: 9 values × 2 lambdas = 18
        gabor_lambdas=None,
        gabor_ksize=5,
        gabor_sigma=10.0,
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

        self.input_shape     = input_shape
        self.use_gabor       = use_gabor
        self.lambda_param    = lambda_param
        self.inh_fraction    = inh_fraction
        self.threshold       = threshold
        self.tau_v           = tau_v
        self.tau_u           = tau_u
        self.input_density   = input_density
        self.input_weight    = input_weight
        self.reservoir_weight= reservoir_weight
        self.receptive_field_size = receptive_field_size
        self.total_reservoir_size = total_reservoir_size
        self.n_reservoirs    = n_reservoirs
        self.reservoir_size  = total_reservoir_size // n_reservoirs
        self.grid_shape      = grid_shape
        self.d_values        = d_values

        assert np.prod(grid_shape) == self.reservoir_size, (
            f"grid_shape {grid_shape} must multiply to reservoir_size "
            f"{self.reservoir_size}"
        )

        # curr_prefac is multiplied into weights (author: curr_prefac = 1/tau_u)
        self.curr_prefac = np.float32(1.0 / tau_u)

        self.rng = np.random.default_rng(seed)
        # The author uses np.random for shuffles inside weight init, so we
        # seed the global numpy RNG for reproducibility.
        np.random.seed(seed)

        # ------------------------------------------------------------------
        # Gabor filter bank
        # ------------------------------------------------------------------
        if use_gabor:
            if gabor_thetas is None:
                # 9 orientations × 2 wavelengths = 18 filters (paper: 18)
                gabor_thetas  = [0, 20, 40, 60, 80, 100, 120, 140, 160]
            if gabor_lambdas is None:
                gabor_lambdas = [5.0, 10.0]

            self.gabor_kernels = build_gabor_filter_bank(
                thetas=gabor_thetas,
                lambdas=gabor_lambdas,
                ksize=gabor_ksize,
                sigma=gabor_sigma,
                gamma=gabor_gamma,
            )  # (n_filters, 2, ksize, ksize)

            n_filters = self.gabor_kernels.shape[0]
            H, W, polarities = input_shape
            self.gabor_out_shape = (H, W, polarities * n_filters)
            self.input_size = int(np.prod(self.gabor_out_shape))
        else:
            self.gabor_kernels   = None
            self.gabor_out_shape = input_shape
            self.input_size      = int(np.prod(input_shape))

        # ------------------------------------------------------------------
        # Build weights for each reservoir
        # ------------------------------------------------------------------
        self.w_in_list  = []
        self.w_rec_list = []

        for d in self.d_values:
            W_in, W_lsm = self._make_weights(d=d)
            self.w_in_list.append(W_in)
            self.w_rec_list.append(W_lsm)

    # -----------------------------------------------------------------------
    # Weight construction
    # -----------------------------------------------------------------------

    def _make_weights(self, d):
        """
        Build receptive-field input weights and distance-biased recurrent
        weights for one reservoir, then scale by curr_prefac.

        Returns W_in as a CSR sparse matrix and W_lsm as a dense float32 array.
        """
        Nx, Ny, Nz = self.grid_shape
        N = self.reservoir_size
        H, W, n_channels = self.gabor_out_shape
        window = self.receptive_field_size
        in_size = self.input_size

        LqWin  = self.input_weight
        LqWlsm = self.reservoir_weight
        lam    = self.lambda_param
        inh_fr = self.inh_fraction

        # ---- Input weights (receptive-field) --------------------------------
        in_conn_range = int(N * self.input_density)

        W_in = np.zeros((in_size, N), dtype=np.float32)

        for i in range(in_size):
            # Decode (ch, y, x) from flat input index
            # Layout: (H, W, n_channels) flattened as H*W*n_channels
            # i = (y * W + x) * n_channels + ch
            ch = i % n_channels
            hw = i // n_channels
            x_in = hw % W
            y_in = hw // W

            # Map to reservoir (x, y) — author uses floor division
            res_x = int((x_in * Nx) / W)
            res_y = int((y_in * Ny) / H)

            # Build window, clamped to [0, Nx) / [0, Ny) — author's logic
            res_x_min = res_x - window // 2
            if res_x_min < 0:
                res_x_min = 0
            res_x_max = res_x_min + window
            if res_x_max > Nx:
                res_x_max = Nx
                res_x_min = Nx - window

            res_y_min = res_y - window // 2
            if res_y_min < 0:
                res_y_min = 0
            res_y_max = res_y_min + window
            if res_y_max > Ny:
                res_y_max = Ny
                res_y_min = Ny - window

            # All neuron indices in window across all Nz slices
            window_locs = []
            for j in range(window):
                row_y = res_y_min + j
                window_locs.append(row_y * Nx + np.arange(res_x_min, res_x_max))
            window_idxs = np.concatenate(window_locs)

            channel_locs = []
            for k in range(Nz):
                channel_locs.append(k * (Nx * Ny) + window_idxs)
            input_perm_i = np.int32(np.concatenate(channel_locs))

            np.random.shuffle(input_perm_i)
            pos_conn = input_perm_i[:in_conn_range]
            neg_conn = input_perm_i[-in_conn_range:]

            W_in[i, pos_conn] =  LqWin
            W_in[i, neg_conn] = -LqWin

        # ---- Recurrent weights (distance-biased, Eq. 5) -------------------
        input_perm = np.arange(N)
        np.random.shuffle(input_perm)
        inh_range = int(inh_fr * N)  # first inh_range indices are inhibitory

        W_lsm = np.zeros((N, N), dtype=np.float32)

        for i in range(N):
            posti = input_perm[i]
            zi = posti // (Nx * Ny)
            yi = (posti - zi * Nx * Ny) // Nx
            xi = (posti - zi * Nx * Ny) % Nx

            for j in range(N):
                prej = input_perm[j]
                zj = prej // (Nx * Ny)
                yj = (prej - zj * Nx * Ny) // Nx
                xj = (prej - zj * Nx * Ny) % Nx

                # Squared Euclidean distance (author uses D, not sqrt(D))
                D_sq = (xi - xj)**2 + (yi - yj)**2 + (zi - zj)**2
                # Eq. 5: P = C * exp(-((sqrt(D) - d) / lambda)^2)
                # Author's MuLRE variant (initWeights_short_long_dist_partition):
                # P2 = C * exp(-((sqrt(D_sq) - d)^2) / lam)
                dist = np.sqrt(D_sq)

                if i < inh_range and j < inh_range:       # II
                    P = 0.3 * np.exp(-((dist - d) ** 2) / lam)
                    if np.random.uniform() < P:
                        W_lsm[prej, posti] = -LqWlsm
                elif i < inh_range and j >= inh_range:    # EI
                    P = 0.1 * np.exp(-((dist - d) ** 2) / lam)
                    if np.random.uniform() < P:
                        W_lsm[prej, posti] =  LqWlsm
                elif i >= inh_range and j < inh_range:    # IE
                    P = 0.05 * np.exp(-((dist - d) ** 2) / lam)
                    if np.random.uniform() < P:
                        W_lsm[prej, posti] = -LqWlsm
                else:                                      # EE
                    P = 0.2 * np.exp(-((dist - d) ** 2) / lam)
                    if np.random.uniform() < P:
                        W_lsm[prej, posti] =  LqWlsm

        np.fill_diagonal(W_lsm, 0.0)

        # Scale by curr_prefac (author: curr_prefac * W before passing to model)
        W_in_scaled  = np.float32(self.curr_prefac * W_in)
        W_lsm_scaled = np.float32(self.curr_prefac * W_lsm)

        # Transpose to match author's convention (W.T for torch nn.Linear compat)
        # Then store W_in as sparse (rows = reservoir neurons, cols = input)
        return csr_matrix(W_in_scaled.T), W_lsm_scaled.T

    # -----------------------------------------------------------------------
    # Preprocessing
    # -----------------------------------------------------------------------

    def _preprocess_sequence(self, x):
        """
        Apply Gabor bank to a sequence of frames.

        Args:
            x: (T, H*W*polarities) float32

        Returns:
            (T, H*W*polarities*n_filters) float32
        """
        if not self.use_gabor:
            return x

        H, W, polarities = self.input_shape
        T = x.shape[0]
        out = []
        for t in range(T):
            frame = x[t].reshape(H, W, polarities)
            gabor_frame = apply_gabor_bank(frame, self.gabor_kernels)
            out.append(gabor_frame.ravel())
        return np.stack(out).astype(np.float32)

    # -----------------------------------------------------------------------
    # Simulation
    # -----------------------------------------------------------------------

    def _run_reservoir(self, x_proc, w_in, w_rec):
        """
        Run a single LIF reservoir and return total spike counts.

        Args:
            x_proc: (T, input_size) float32
            w_in:   CSR sparse (reservoir_size, input_size)
            w_rec:  dense float32 (reservoir_size, reservoir_size)

        Returns:
            spike_counts: (reservoir_size,) float32
        """
        N  = self.reservoir_size
        v  = np.zeros(N, dtype=np.float32)
        u  = np.zeros(N, dtype=np.float32)
        spk= np.zeros(N, dtype=np.float32)
        spike_counts = np.zeros(N, dtype=np.float32)

        # Author: alpha = exp(-1/tau_u), beta = 1 - 1/tau_v
        alpha   = np.float32(np.exp(-1.0 / self.tau_u))   # synaptic current decay
        beta    = np.float32(1.0 - 1.0 / self.tau_v)      # membrane decay

        # Pre-project all inputs: (T, N)  — w_in is (N, input_size)
        input_proj = np.asarray(w_in.dot(x_proc.T).T, dtype=np.float32)  # (T, N)

        for t in range(x_proc.shape[0]):
            u   = alpha * u + input_proj[t] + spk @ w_rec
            v   = beta  * v + u
            spk = (v >= self.threshold).astype(np.float32)
            v[spk > 0] = 0.0
            spike_counts += spk

        return spike_counts

    def transform_one(self, x):
        """
        Transform a single sample.

        Args:
            x: (T, H*W*polarities)

        Returns:
            (total_reservoir_size,) float32
        """
        x_proc = self._preprocess_sequence(x)
        return np.concatenate([
            self._run_reservoir(x_proc, w_in, w_rec)
            for w_in, w_rec in zip(self.w_in_list, self.w_rec_list)
        ])

    def transform(self, X, chunk_size=50):
        """
        Transform a batch of samples.

        Args:
            X:          (N, T, H*W*polarities)
            chunk_size: number of samples to preprocess at once (memory cap)

        Returns:
            (N, total_reservoir_size) float32
        """
        features = []
        total = len(X)
        for start in range(0, total, chunk_size):
            end = min(start + chunk_size, total)
            for i, x in enumerate(X[start:end]):
                features.append(self.transform_one(x))
            print(f"  Transformed {end}/{total} samples", flush=True)
        return np.stack(features)