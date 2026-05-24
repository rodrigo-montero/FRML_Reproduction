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

from src.datasets.event_utils import shd_events_to_bins
from src.models.tepre import TEPRE


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


def load_subset(train=True, n_samples=1000, n_time_bins=1000, input_size=700, seed=42):
    dataset = tonic.datasets.SHD(
        save_to=str(DATA_PATH),
        train=train
    )

    rng = np.random.default_rng(seed)
    indices = rng.choice(len(dataset), size=min(n_samples, len(dataset)), replace=False)

    X, y = [], []
    for count, i in enumerate(indices):
        events, label = dataset[int(i)]
        frames = shd_events_to_bins(
            events,
            n_time_bins=n_time_bins,
            input_size=input_size,
        )
        X.append(frames)
        y.append(label)
        if (count + 1) % 100 == 0:
            print(f"Loaded {count + 1}/{len(indices)} samples")

    return np.stack(X), np.array(y)


def main():
    # ---- Dataset config ----
    n_time_bins = 1000
    input_size  = 700
    train_size  = 8156
    test_size   = 2264

    # ---- Paper-specified hyperparameters ----
    total_reservoir_size = 3000
    n_partitions         = 6
    # Paper: Nx=Ny=10, Nz=30; with 6 partitions each partition is 10×10×5=500
    grid_shape           = (10, 10, 5)
    threshold            = 20.0
    tau_v                = 40.0   # paper: (tau_v, tau_u) = (40, 20) for SHD
    tau_u                = 20.0
    reservoir_weight     = 1.0

    # ---- Author's code hyperparameters (not in paper) ----
    lambda_param          = 9.0
    inh_fraction          = 0.2
    input_density         = 0.02
    input_weight          = 1.0
    inter_partition_weight= 1.0   # magnitude; always applied as inhibitory

    print("Loading SHD train subset...")
    X_train, y_train = load_subset(train=True,  n_samples=train_size, n_time_bins=n_time_bins, input_size=input_size, seed=SEED)

    print("Loading SHD test subset...")
    X_test,  y_test  = load_subset(train=False, n_samples=test_size,  n_time_bins=n_time_bins, input_size=input_size, seed=SEED)

    print("Train labels:", np.bincount(y_train))
    print("Test labels: ", np.bincount(y_test))
    print("Input shape: ", X_train.shape)

    print("Creating SHD TEPRE...")
    tepre = TEPRE(
        input_size=input_size,
        total_reservoir_size=total_reservoir_size,
        n_partitions=n_partitions,
        grid_shape=grid_shape,
        input_density=input_density,
        input_weight=input_weight,
        reservoir_weight=reservoir_weight,
        inter_partition_weight=inter_partition_weight,
        lambda_param=lambda_param,
        inh_fraction=inh_fraction,
        threshold=threshold,
        tau_v=tau_v,
        tau_u=tau_u,
        seed=SEED,
    )

    print("Transforming train data...")
    Z_train = tepre.transform(X_train)

    print("Transforming test data...")
    Z_test = tepre.transform(X_test)

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
            C=1.0,
            random_state=SEED,
        )
    )
    clf.fit(Z_train, y_train)

    preds = clf.predict(Z_test)
    acc   = accuracy_score(y_test, preds)
    print(f"SHD TEPRE accuracy: {acc * 100:.2f}%")

    results = {
        "experiment":    "shd-tepre",
        "accuracy":      float(acc),
        "train_samples": int(len(y_train)),
        "test_samples":  int(len(y_test)),
        "seed":          SEED,
        "timestamp":     datetime.now().isoformat(),
        "model": {
            "total_reservoir_size":  total_reservoir_size,
            "n_partitions":          n_partitions,
            "tau_v":                 tau_v,
            "tau_u":                 tau_u,
            "threshold":             threshold,
            "lambda_param":          lambda_param,
            "inh_fraction":          inh_fraction,
            "input_density":         input_density,
            "input_weight":          input_weight,
            "inter_partition_weight":inter_partition_weight,
        }
    }

    save_results(results, "shd_tepre.json")
    np.savez(
        PROJECT_ROOT / "results" / "shd_tepre_predictions.npz",
        y_true=y_test,
        y_pred=preds,
    )


if __name__ == "__main__":
    main()