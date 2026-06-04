import numpy as np
from scipy.sparse import csr_matrix


class TEPRE:
    """
    Temporal Excitation Partitioned Reservoir Ensemble (TEPRE).

    Paper-specified values are faithfully reproduced.  Values that the paper
    leaves unspecified are taken from the author's reference implementation
    (lsm_weight_definitions.py / lsm_models.py).

    Paper:
        - Each partition reservoir receives input only during its time window
        - Sparse inhibitory cross-partition connections
        - P(i,j) = C * exp(-(D(i,j)/lambda)^2)   [Eq. 4, standard LSM]
        - C: EE=0.2, EI=0.1, IE=0.05, II=0.3
        - wlsm = 1; theta = 20; tau_u = tau_v = 16 (N-MNIST), (40,20) (SHD)
        - Standard (non-receptive-field) input for TEPRE

    Author's code (unspecified in paper):
        - lambda = 9
        - D is squared Euclidean distance, probability = C * exp(-D_sq / lam)
          (author's initWeights_partition uses `np.exp(-D/lam)` where D is D_sq)
        - inhibitory fraction = 0.2 (first 20 % of a random permutation)
        - input connections: exactly floor(partition_N * density) positive
          and the same number negative, drawn from random shuffle
        - the SAME W_lsm_part is reused for every partition (author builds
          one partition weight matrix and tiles it block-diagonally)
        - cross-partition inhibition: deterministic shift
            W_long[i, (i + partition_N) % N] = -LqWlsm_long  for all i
          (initWeights_partition_cross_partition_inh)
        - curr_prefac = 1 / tau_u multiplied into W_in and W_lsm
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
        inter_partition_weight=1.0,    # magnitude; always inhibitory (negated)
        lambda_param=9.0,              # author uses 9
        inh_fraction=0.2,              # author uses 0.2
        threshold=20.0,
        tau_v=16.0,
        tau_u=16.0,
        seed=42,
    ):
        assert total_reservoir_size % n_partitions == 0, (
            "total_reservoir_size must be divisible by n_partitions"
        )

        self.input_size           = input_size
        self.total_reservoir_size = total_reservoir_size
        self.n_partitions         = n_partitions
        self.partition_size       = total_reservoir_size // n_partitions
        self.grid_shape           = grid_shape
        self.lambda_param         = lambda_param
        self.inh_fraction         = inh_fraction
        self.threshold            = threshold
        self.tau_v                = tau_v
        self.tau_u                = tau_u
        self.input_density        = input_density
        self.input_weight         = input_weight
        self.reservoir_weight     = reservoir_weight
        self.inter_partition_weight = inter_partition_weight

        assert np.prod(grid_shape) == self.partition_size, (
            f"grid_shape {grid_shape} must multiply to partition_size "
            f"{self.partition_size}"
        )

        # curr_prefac is multiplied into weights (author: 1/tau_u)
        self.curr_prefac = np.float32(1.0 / tau_u)

        np.random.seed(seed)

        # ------------------------------------------------------------------
        # Build weights
        # ------------------------------------------------------------------
        W_ins_raw, W_lsm_raw, W_long_raw = self._make_all_weights()

        # Scale by curr_prefac (author applies this before passing to model)
        self.W_ins  = [np.float32(self.curr_prefac * w) for w in W_ins_raw]
        W_lsm_total = np.float32(self.curr_prefac * (W_lsm_raw + W_long_raw))

        # Transpose to (N_post, N_pre) for matmul: spk @ W_rec
        self.W_rec = W_lsm_total.T   # (N, N)

    # -----------------------------------------------------------------------
    # Weight construction
    # -----------------------------------------------------------------------

    def _make_all_weights(self):
        """
        Build input, recurrent, and cross-partition inhibitory weights.

        Follows initWeights_partition_cross_partition_inh from the author's
        lsm_weight_definitions.py.

        Returns:
            W_ins:   list of n_partitions arrays (in_size, N)
            W_lsm:   (N, N) block-diagonal recurrent matrix
            W_long:  (N, N) cross-partition inhibitory shift matrix
        """
        Nx, Ny, Nz = self.grid_shape
        N           = self.total_reservoir_size
        partition_N = self.partition_size
        LqWin       = self.input_weight
        LqWlsm      = self.reservoir_weight
        LqWlsm_long = self.inter_partition_weight
        lam         = self.lambda_param
        inh_fr      = self.inh_fraction
        in_size     = self.input_size
        in_conn_range = int(partition_N * self.input_density)

        # ---- Build one shared W_in_part then tile it per partition ----------
        W_in_part = np.zeros((in_size, partition_N), dtype=np.float32)
        for i in range(in_size):
            perm = np.arange(partition_N)
            np.random.shuffle(perm)
            pos_conn = perm[:in_conn_range]
            neg_conn = perm[-in_conn_range:]
            W_in_part[i, pos_conn] =  LqWin
            W_in_part[i, neg_conn] = -LqWin

        W_ins = []
        for part in range(self.n_partitions):
            W_in = np.zeros((in_size, N), dtype=np.float32)
            W_in[:, part * partition_N:(part + 1) * partition_N] = W_in_part
            W_ins.append(W_in)

        # ---- Build one shared W_lsm_part then tile block-diagonally --------
        input_perm = np.arange(partition_N)
        np.random.shuffle(input_perm)
        inh_range = int(inh_fr * partition_N)

        Nz_part = Nz // self.n_partitions
        W_lsm_part = np.zeros((partition_N, partition_N), dtype=np.float32)

        for i in range(partition_N):
            posti = input_perm[i]
            zi = posti // (Nx * Ny)
            yi = (posti - zi * Nx * Ny) // Nx
            xi = (posti - zi * Nx * Ny) % Nx

            for j in range(partition_N):
                prej = input_perm[j]
                zj = prej // (Nx * Ny)
                yj = (prej - zj * Nx * Ny) // Nx
                xj = (prej - zj * Nx * Ny) % Nx

                # Author uses squared distance directly as D, then exp(-D/lam)
                D = (xi - xj)**2 + (yi - yj)**2 + (zi - zj)**2

                if i < inh_range and j < inh_range:        # II
                    P = 0.3 * np.exp(-D / lam)
                    if np.random.uniform() < P:
                        W_lsm_part[prej, posti] = -LqWlsm
                elif i < inh_range and j >= inh_range:     # EI
                    P = 0.1 * np.exp(-D / lam)
                    if np.random.uniform() < P:
                        W_lsm_part[prej, posti] =  LqWlsm
                elif i >= inh_range and j < inh_range:     # IE
                    P = 0.05 * np.exp(-D / lam)
                    if np.random.uniform() < P:
                        W_lsm_part[prej, posti] = -LqWlsm
                else:                                       # EE
                    P = 0.2 * np.exp(-D / lam)
                    if np.random.uniform() < P:
                        W_lsm_part[prej, posti] =  LqWlsm

        np.fill_diagonal(W_lsm_part, 0.0)

        # Tile into full block-diagonal matrix
        W_lsm = np.zeros((N, N), dtype=np.float32)
        for part in range(self.n_partitions):
            s = part * partition_N
            e = s + partition_N
            W_lsm[s:e, s:e] = W_lsm_part

        # ---- Cross-partition inhibition: deterministic shift ----------------
        # Author: W_lsm_long[i, (i + partition_N) % N] = -LqWlsm_long  ∀i
        W_long = np.zeros((N, N), dtype=np.float32)
        for i in range(N):
            W_long[i, (i + partition_N) % N] = -LqWlsm_long

        return W_ins, W_lsm, W_long

    # -----------------------------------------------------------------------
    # Simulation
    # -----------------------------------------------------------------------

    def transform_one(self, x):
        """
        Run the TEPRE on a single sample.

        Args:
            x: (T, input_size) float32

        Returns:
            spike_counts: (total_reservoir_size,) float32
        """
        T  = x.shape[0]
        N  = self.total_reservoir_size
        partition_N    = self.partition_size
        partition_steps = T // self.n_partitions

        v  = np.zeros(N, dtype=np.float32)
        u  = np.zeros(N, dtype=np.float32)
        spk= np.zeros(N, dtype=np.float32)
        spike_counts = np.zeros(N, dtype=np.float32)

        # Author: alpha = exp(-1/tau_u), beta = 1 - 1/tau_v
        alpha = np.float32(np.exp(-1.0 / self.tau_u))
        beta  = np.float32(1.0 - 1.0 / self.tau_v)

        # Pre-project inputs for each partition: (T, N)
        # Each W_ins[p] is (in_size, N) → (x @ W_ins[p].T) is not right;
        # W_ins[p] shape is (in_size, N), so input_proj = x @ W_ins[p]  (T,N)
        # Author transposes for torch Linear compat; here we keep (in_size, N)
        # and do x[t] @ W_in  → (N,)
        # Pre-compute all at once per partition:
        input_projs = []
        for p in range(self.n_partitions):
            proj = x @ self.W_ins[p]  # (T, N)
            input_projs.append(proj.astype(np.float32))

        Win_ind = 0
        for t in range(T):
            if t % partition_steps == 0 and Win_ind < self.n_partitions:
                current_proj = input_projs[Win_ind]
                Win_ind = min(Win_ind + 1, self.n_partitions - 1)

            inp = current_proj[t]

            # cross-partition: author adds long_inh(spk[t - partition_steps])
            if t >= partition_steps:
                # W_long is already incorporated into W_rec; no extra term needed
                pass

            u    = alpha * u + inp + spk @ self.W_rec
            v    = beta  * v + u
            spk  = (v >= self.threshold).astype(np.float32)
            v[spk > 0] = 0.0
            spike_counts += spk

        return spike_counts

    def transform(self, X):
        """
        Args:
            X: (n_samples, T, input_size)

        Returns:
            (n_samples, total_reservoir_size) float32
        """
        features = []
        for i, x in enumerate(X):
            features.append(self.transform_one(x))
            if (i + 1) % 100 == 0:
                print(f"  Transformed {i + 1}/{len(X)} samples", flush=True)
        return np.stack(features)