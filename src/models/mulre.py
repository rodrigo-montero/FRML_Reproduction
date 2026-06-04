import numpy as np
from scipy.sparse import csr_matrix

def build_gabor_filter_bank(thetas, lambdas, ksize=5, sigma=10.0, gamma=0.5):
    """
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
    phi = np.pi / 2
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

            norm = np.linalg.norm(kernel)
            if norm > 0:
                kernel = kernel / norm
            kernel = kernel.astype(np.float32)

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

    x_pol = x_frame.transpose(2, 0, 1).astype(np.float32)
    X_f   = rfft2(x_pol, s=(H, W))

    out = np.empty((n_filters, polarities, H, W), dtype=np.float32)
    for f in range(n_filters):
        for p in range(polarities):
            k_pad = np.zeros((H, W), dtype=np.float32)
            kh, kw = kernels[f, p].shape
            k_pad[:kh, :kw] = kernels[f, p]
            K_f = rfft2(k_pad, s=(H, W))
            resp = np.real(np.fft.irfft2(X_f[p] * K_f, s=(H, W))).astype(np.float32)
            np.maximum(resp, 0.0, out=resp)
            out[f, p] = resp

    out = out.transpose(2, 3, 1, 0)
    return out.reshape(H, W, polarities * n_filters)


class MuLRE:
    """
    Multi-Length Scale Reservoir Ensemble (MuLRE).

    Paper:
        - 3 reservoirs with d = {0, 4, 6} (2-reservoir: {0, 5})
        - P(i,j) = C * exp(-((D(i,j) - d) / lambda)^2)   [Eq. 5]
        - C: EE=0.2, EI=0.1, IE=0.05, II=0.3
        - wlsm = 1  (excitatory +1, inhibitory -1)
        - theta = 20, tau_u = tau_v = 16  (N-MNIST)
        - 3-D reservoir grid, receptive-field input, 18 Gabor filters

    Author's code (unspecified in paper):
        - lambda = 9
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
        lambda_param=9.0,
        inh_fraction=0.2,
        threshold=20.0,
        tau_v=16.0,
        tau_u=16.0,
        seed=42,
        use_gabor=True,
        gabor_thetas=None,
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
        self.curr_prefac = np.float32(1.0 / tau_u)
        self.rng = np.random.default_rng(seed)
        np.random.seed(seed)

        if use_gabor:
            if gabor_thetas is None:
                gabor_thetas  = [0, 20, 40, 60, 80, 100, 120, 140, 160]
            if gabor_lambdas is None:
                gabor_lambdas = [5.0, 10.0]

            self.gabor_kernels = build_gabor_filter_bank(
                thetas=gabor_thetas,
                lambdas=gabor_lambdas,
                ksize=gabor_ksize,
                sigma=gabor_sigma,
                gamma=gabor_gamma,
            )

            n_filters = self.gabor_kernels.shape[0]
            H, W, polarities = input_shape
            self.gabor_out_shape = (H, W, polarities * n_filters)
            self.input_size = int(np.prod(self.gabor_out_shape))
        else:
            self.gabor_kernels   = None
            self.gabor_out_shape = input_shape
            self.input_size      = int(np.prod(input_shape))

        self.w_in_list  = []
        self.w_rec_list = []

        for d in self.d_values:
            W_in, W_lsm = self._make_weights(d=d)
            self.w_in_list.append(W_in)
            self.w_rec_list.append(W_lsm)

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

        in_conn_range = int(N * self.input_density)

        W_in = np.zeros((in_size, N), dtype=np.float32)

        for i in range(in_size):
            ch = i % n_channels
            hw = i // n_channels
            x_in = hw % W
            y_in = hw // W

            res_x = int((x_in * Nx) / W)
            res_y = int((y_in * Ny) / H)

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

        input_perm = np.arange(N)
        np.random.shuffle(input_perm)
        inh_range = int(inh_fr * N)

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

                D_sq = (xi - xj)**2 + (yi - yj)**2 + (zi - zj)**2
                dist = np.sqrt(D_sq)

                if i < inh_range and j < inh_range:
                    P = 0.3 * (np.exp(-((dist - d)) / lam) ** 2)
                    if np.random.uniform() < P:
                        W_lsm[prej, posti] = -LqWlsm
                elif i < inh_range and j >= inh_range:
                    P = 0.1 * (np.exp(-((dist - d)) / lam) ** 2)
                    if np.random.uniform() < P:
                        W_lsm[prej, posti] =  LqWlsm
                elif i >= inh_range and j < inh_range:
                    P = 0.05 * (np.exp(-((dist - d)) / lam) ** 2)
                    if np.random.uniform() < P:
                        W_lsm[prej, posti] = -LqWlsm
                else:
                    P = 0.2 * (np.exp(-((dist - d)) / lam) ** 2)
                    if np.random.uniform() < P:
                        W_lsm[prej, posti] =  LqWlsm

        np.fill_diagonal(W_lsm, 0.0)

        W_in_scaled  = np.float32(self.curr_prefac * W_in)
        W_lsm_scaled = np.float32(self.curr_prefac * W_lsm)

        return csr_matrix(W_in_scaled.T), W_lsm_scaled.T

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

        alpha   = np.float32(np.exp(-1.0 / self.tau_u))
        beta    = np.float32(1.0 - 1.0 / self.tau_v)

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