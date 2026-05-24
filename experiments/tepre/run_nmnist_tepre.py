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


def save_results(results, filename):
    results_dir = PROJECT_ROOT / "results" / "nmnist_tepre"
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
    indices = rng.choice(len(dataset), size=n_samples, replace=False)

    X, y = [], []
    for count, i in enumerate(indices):
        events, label = dataset[int(i)]
        frames = events_to_frames(events, n_time_bins=n_time_bins)
        X.append(frames)
        y.append(label)
        if (count + 1) % 100 == 0:
            print(f"  Loaded {count + 1}/{n_samples} samples")

    return np.stack(X), np.array(y)


def run_seed(seed, X_train, y_train, X_test, y_test):
    """Full pipeline for one seed: build model, transform, classify."""
    random.seed(seed)
    np.random.seed(seed)

    input_size = X_train.shape[-1]

    # ---- Paper-specified hyperparameters ----
    total_reservoir_size = 3600
    n_partitions         = 3
    grid_shape           = (10, 10, 12)
    threshold            = 20.0
    tau_v                = 16.0
    tau_u                = 16.0
    reservoir_weight     = 1.0

    # ---- Author's code hyperparameters (not in paper) ----
    lambda_param          = 9.0
    inh_fraction          = 0.2
    input_density         = 0.02
    input_weight          = 1.0
    inter_partition_weight= 1.0

    print(f"\n  Building TEPRE (seed={seed})...")
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
        seed=seed,
    )

    print(f"  Transforming train data (seed={seed})...")
    Z_train = tepre.transform(X_train)

    print(f"  Transforming test data (seed={seed})...")
    Z_test = tepre.transform(X_test)

    print(f"  Training classifier (seed={seed})...")
    clf = SGDClassifier(max_iter=10000, tol=1e-6, random_state=seed)
    clf.fit(Z_train, y_train)

    preds = clf.predict(Z_test)
    acc   = accuracy_score(y_test, preds)

    print(f"  Seed {seed} accuracy: {acc * 100:.2f}%")

    return {
        "seed":               seed,
        "accuracy":           float(acc),
        "predictions":        preds,
    }


def main():
    n_time_bins = 90
    # train_size  = 60000
    # test_size   = 10000
    # For testing!!!
    train_size = 10
    test_size = 10
    # ---- Multi-seed runs ----
    all_results = []
    for seed in SEEDS:
        print(f"\n{'='*55}")
        print(f"  SEED {seed}")
        print(f"{'='*55}")
        print("Loading train subset...")
        X_train, y_train = load_subset(train=True, n_samples=train_size, n_time_bins=n_time_bins, seed=42)

        print("Loading test subset...")
        X_test, y_test = load_subset(train=False, n_samples=test_size, n_time_bins=n_time_bins, seed=42)

        print(f"Train labels: {np.bincount(y_train, minlength=10)}")
        print(f"Test labels:  {np.bincount(y_test, minlength=10)}")
        print(f"Input size:   {X_train.shape[-1]}")
        result = run_seed(seed, X_train, y_train, X_test, y_test)
        all_results.append(result)

    # ---- Aggregate statistics ----
    accuracies = [r["accuracy"] for r in all_results]
    mean_acc   = float(np.mean(accuracies))
    std_acc    = float(np.std(accuracies))
    min_acc    = float(np.min(accuracies))
    max_acc    = float(np.max(accuracies))
    best_seed  = SEEDS[int(np.argmax(accuracies))]

    print(f"\n{'='*55}")
    print(f"  MULTI-SEED SUMMARY  (N-MNIST TEPRE)")
    print(f"{'='*55}")
    for r in all_results:
        print(f"  Seed {r['seed']:>3d}:  {r['accuracy']*100:.2f}%")
    print(f"  -----------------------------------------------")
    print(f"  Mean ± Std:  {mean_acc*100:.2f}% ± {std_acc*100:.2f}%")
    print(f"  Min / Max:   {min_acc*100:.2f}% / {max_acc*100:.2f}%")
    print(f"  Best seed:   {best_seed}")

    # ---- Save ----
    summary = {
        "experiment": "nmnist-tepre-multiseed",
        "seeds":      SEEDS,
        "timestamp":  datetime.now().isoformat(),
        "per_seed": [
            {
                "seed":               r["seed"],
                "accuracy":           r["accuracy"],
            }
            for r in all_results
        ],
        "aggregate": {
            "mean_accuracy":           mean_acc,
            "std_accuracy":            std_acc,
            "min_accuracy":            min_acc,
            "max_accuracy":            max_acc,
            "best_seed":               best_seed,
        },
        "dataset": {
            "train_samples": int(len(y_train)),
            "test_samples":  int(len(y_test)),
            "n_time_bins":   n_time_bins,
        },
        "model": {
            "total_reservoir_size":  3600,
            "n_partitions":          3,
            "tau_v":                 16.0,
            "tau_u":                 16.0,
            "threshold":             20.0,
            "lambda_param":          9.0,
            "inh_fraction":          0.2,
            "input_density":         0.02,
            "input_weight":          1.0,
            "inter_partition_weight":1.0,
        },
    }

    save_results(summary, "nmnist_tepre_multiseed.json")

    for r in all_results:
        np.savez(
            PROJECT_ROOT / "results" / "nmnist_tepre" / f"nmnist_tepre_seed{r['seed']}_predictions.npz",
            y_true=y_test,
            y_pred=r["predictions"],
        )

    print("\nAll results saved.")


if __name__ == "__main__":
    main()