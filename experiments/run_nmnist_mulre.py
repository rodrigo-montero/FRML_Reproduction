import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import numpy as np
import tonic
import random
import json
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.datasets.event_utils import events_to_frames
from src.models.mulre import MuLRE


DATA_PATH = PROJECT_ROOT / "data" / "raw"

SEED = 42

random.seed(SEED)
np.random.seed(SEED)


def save_results(results, filename):
    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    path = results_dir / filename
    with open(path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Saved results to {path}")


def load_subset(train=True, n_samples=1000, n_time_bins=90, seed=42):
    dataset = tonic.datasets.NMNIST(
        save_to=str(DATA_PATH),
        train=train
    )

    rng = np.random.default_rng(seed)
    indices = rng.choice(len(dataset), size=min(n_samples, len(dataset)), replace=False)

    X, y = [], []
    for count, i in enumerate(indices):
        events, label = dataset[int(i)]
        frames = events_to_frames(
            events,
            sensor_size=(34, 34, 2),
            n_time_bins=n_time_bins
        )
        X.append(frames)
        y.append(label)
        if (count + 1) % 100 == 0:
            print(f"Loaded {count + 1}/{len(indices)} samples")

    return np.stack(X), np.array(y)


def main():
    # ---- Dataset config ----
    n_time_bins = 90
    train_size  = 60000
    test_size   = 10000
    input_shape = (34, 34, 2)

    # ---- Paper-specified hyperparameters ----
    total_reservoir_size = 3600
    n_reservoirs         = 3
    grid_shape           = (10, 10, 12)   # 10*10*12 = 1200 per reservoir
    d_values             = [0, 4, 6]      # paper: 3-reservoir MuLRE
    receptive_field_size = 6              # paper: "5 or 6"
    threshold            = 20.0           # paper: theta = 20
    tau_v                = 16.0           # paper: tau_v = 16 for N-MNIST
    tau_u                = 16.0           # paper: tau_u = 16 for N-MNIST
    reservoir_weight     = 1.0            # paper: w_lsm = 1

    # ---- Author's code hyperparameters (not in paper) ----
    lambda_param     = 9.0    # author's initWeights calls use lam=9
    inh_fraction     = 0.2    # author: inh_fr=0.2
    input_density    = 0.02   # author: in_conn_density=0.02 (approx)
    input_weight     = 1.0    # author: LqWin=1 (before curr_prefac scaling)

    # Gabor: author uses cv2.getGaborKernel, ksize=5, sigma=10, gamma=0.5, phi=pi/2
    # 9 theta values × 2 lambda values = 18 filters (paper: 18 Gabor filters)
    gabor_thetas  = [0, 20, 40, 60, 80, 100, 120, 140, 160]
    gabor_lambdas = [5.0, 10.0]
    gabor_ksize   = 5
    gabor_sigma   = 10.0
    gabor_gamma   = 0.5

    print("Loading N-MNIST train subset...")
    X_train, y_train = load_subset(train=True,  n_samples=train_size, n_time_bins=n_time_bins, seed=SEED)

    print("Loading N-MNIST test subset...")
    X_test,  y_test  = load_subset(train=False, n_samples=test_size,  n_time_bins=n_time_bins, seed=SEED)

    print("Train labels:", np.bincount(y_train, minlength=10))
    print("Test labels: ", np.bincount(y_test,  minlength=10))
    print("Input shape: ", X_train.shape)

    print("Creating MuLRE...")
    mulre = MuLRE(
        input_shape=input_shape,
        total_reservoir_size=total_reservoir_size,
        n_reservoirs=n_reservoirs,
        grid_shape=grid_shape,
        d_values=d_values,
        receptive_field_size=receptive_field_size,
        input_density=input_density,
        input_weight=input_weight,
        reservoir_weight=reservoir_weight,
        lambda_param=lambda_param,
        inh_fraction=inh_fraction,
        threshold=threshold,
        tau_v=tau_v,
        tau_u=tau_u,
        seed=SEED,
        use_gabor=True,
        gabor_thetas=gabor_thetas,
        gabor_lambdas=gabor_lambdas,
        gabor_ksize=gabor_ksize,
        gabor_sigma=gabor_sigma,
        gabor_gamma=gabor_gamma,
    )

    print("Transforming train data...")
    Z_train = mulre.transform(X_train)

    print("Transforming test data...")
    Z_test = mulre.transform(X_test)

    print("Feature shape train:", Z_train.shape)
    print("Feature shape test: ", Z_test.shape)
    print("Mean train feature activity:", Z_train.mean())
    print("Max  train feature activity:", Z_train.max())
    print("Nonzero fraction (train):   ", (Z_train > 0).mean())

    print("Training linear classifier...")
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=3000,
            n_jobs=-1,
            solver="lbfgs",
            random_state=SEED,
        )
    )
    clf.fit(Z_train, y_train)

    preds = clf.predict(Z_test)
    acc   = accuracy_score(y_test, preds)
    print(f"N-MNIST MuLRE accuracy: {acc * 100:.2f}%")

    results = {
        "experiment":    "nmnist-mulre",
        "accuracy":      float(acc),
        "train_samples": int(len(y_train)),
        "test_samples":  int(len(y_test)),
        "seed":          SEED,
        "timestamp":     datetime.now().isoformat(),
        "model": {
            "total_reservoir_size": total_reservoir_size,
            "n_reservoirs":         n_reservoirs,
            "d_values":             d_values,
            "tau_v":                tau_v,
            "tau_u":                tau_u,
            "threshold":            threshold,
            "lambda_param":         lambda_param,
            "inh_fraction":         inh_fraction,
            "input_density":        input_density,
            "input_weight":         input_weight,
            "receptive_field_size": receptive_field_size,
        }
    }

    save_results(results, "nmnist_mulre.json")
    np.savez(
        PROJECT_ROOT / "results" / "nmnist_mulre_predictions.npz",
        y_true=y_test,
        y_pred=preds,
    )


if __name__ == "__main__":
    main()