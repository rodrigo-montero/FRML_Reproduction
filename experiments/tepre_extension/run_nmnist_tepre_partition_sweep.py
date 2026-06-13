import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

import numpy as np
import tonic
import random
import json
from datetime import datetime
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score

from src.datasets.event_utils import events_to_frames
from src.models.tepre import TEPRE


DATA_PATH = PROJECT_ROOT / "data" / "raw"

SEEDS = [21, 42, 67]

GRID_SHAPES = {
    1: (10, 10, 36),   # partition_size = 3600
    2: (10, 10, 18),   # partition_size = 1800
    3: (10, 10, 12),   # partition_size = 1200 — same as teammates
    4: (10, 10, 9),    # partition_size = 900
    5: (10, 12, 6),    # partition_size = 720
    6: (10, 10, 6),    # partition_size = 600
}


def save_results(results, filename):
    results_dir = PROJECT_ROOT / "results" / "nmnist_tepre_partition_sweep"
    results_dir.mkdir(exist_ok=True)
    path = results_dir / filename
    with open(path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Saved results to {path}")


def load_data(n_time_bins=90, train_size=60000, test_size=10000):
    dataset_train = tonic.datasets.NMNIST(save_to=str(DATA_PATH), train=True)
    dataset_test  = tonic.datasets.NMNIST(save_to=str(DATA_PATH), train=False)

    rng = np.random.default_rng(42)

    def load_subset(dataset, n_samples):
        indices = rng.choice(len(dataset), size=min(n_samples, len(dataset)), replace=False)
        X, y = [], []
        for count, i in enumerate(indices):
            events, label = dataset[int(i)]
            frames = events_to_frames(events, n_time_bins=n_time_bins)
            X.append(frames)
            y.append(label)
            if (count + 1) % 1000 == 0:
                print(f"  Loaded {count + 1}/{n_samples} samples")
        return np.stack(X), np.array(y)

    print("Loading train data...")
    X_train, y_train = load_subset(dataset_train, train_size)
    print("Loading test data...")
    X_test, y_test = load_subset(dataset_test, test_size)

    return X_train, y_train, X_test, y_test


def run_one(seed, n_partitions, X_train, y_train, X_test, y_test):
    random.seed(seed)
    np.random.seed(seed)

    total_reservoir_size = 3600
    grid_shape = GRID_SHAPES[n_partitions]

    print(f"\n  n_partitions={n_partitions}, seed={seed}, grid_shape={grid_shape}")

    tepre = TEPRE(
        input_size=X_train.shape[-1],
        total_reservoir_size=total_reservoir_size,
        n_partitions=n_partitions,
        grid_shape=grid_shape,
        input_density=0.02,
        input_weight=1.0,
        reservoir_weight=1.0,
        inter_partition_weight=1.0,
        lambda_param=9.0,
        inh_fraction=0.2,
        threshold=20.0,
        tau_v=16.0,
        tau_u=16.0,
        seed=seed,
    )

    print(f"  Transforming train data...")
    Z_train = tepre.transform(X_train)
    print(f"  Transforming test data...")
    Z_test = tepre.transform(X_test)

    print(f"  Training classifier...")
    clf = SGDClassifier(max_iter=10000, tol=1e-6, random_state=seed)
    clf.fit(Z_train, y_train)

    preds = clf.predict(Z_test)
    acc = accuracy_score(y_test, preds)
    print(f"  n_partitions={n_partitions}, seed={seed} -> accuracy: {acc * 100:.2f}%")

    return float(acc), preds


def main(n_partitions):
    X_train, y_train, X_test, y_test = load_data()

    partition_results = []
    all_preds = []

    for seed in SEEDS:
        acc, preds = run_one(seed, n_partitions, X_train, y_train, X_test, y_test)
        partition_results.append({"seed": seed, "accuracy": acc})
        all_preds.append(preds)

    accuracies = [r["accuracy"] for r in partition_results]
    summary = {
        "experiment": "nmnist-tepre-partition-sweep",
        "n_partitions": n_partitions,
        "seeds": SEEDS,
        "timestamp": datetime.now().isoformat(),
        "per_seed": partition_results,
        "mean_accuracy": float(np.mean(accuracies)),
        "std_accuracy": float(np.std(accuracies)),
        "min_accuracy": float(np.min(accuracies)),
        "max_accuracy": float(np.max(accuracies)),
        "dataset": {
            "train_samples": int(len(y_train)),
            "test_samples": int(len(y_test)),
            "n_time_bins": 90,
        },
        "model": {
            "total_reservoir_size": 3600,
            "grid_shape": list(GRID_SHAPES[n_partitions]),
            "tau_v": 16.0,
            "tau_u": 16.0,
            "threshold": 20.0,
            "lambda_param": 9.0,
            "inh_fraction": 0.2,
            "input_density": 0.02,
            "input_weight": 1.0,
            "inter_partition_weight": 1.0,
        },
    }

    print(f"\n  n_partitions={n_partitions}: mean={np.mean(accuracies)*100:.2f}% std={np.std(accuracies)*100:.2f}%")

    save_results(summary, f"partitions_{n_partitions}.json")

    results_dir = PROJECT_ROOT / "results" / "nmnist_tepre_partition_sweep"
    for i, seed in enumerate(SEEDS):
        np.savez(
            results_dir / f"partitions_{n_partitions}_seed{seed}_predictions.npz",
            y_true=y_test,
            y_pred=all_preds[i],
        )

    print("Done. All results saved.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python run_nmnist_tepre_partition_sweep.py <n_partitions>")
        sys.exit(1)
    main(int(sys.argv[1]))