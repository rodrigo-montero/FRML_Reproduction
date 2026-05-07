import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import numpy as np
import tonic
import random
import json
from datetime import datetime
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from src.datasets.event_utils import events_to_frames
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



def load_subset(train=True, n_samples=1000, n_time_bins=90, seed=42):
    dataset = tonic.datasets.NMNIST(
        save_to=str(DATA_PATH),
        train=train
    )

    rng = np.random.default_rng(seed)
    indices = rng.choice(len(dataset), size=n_samples, replace=False)

    X = []
    y = []

    for count, i in enumerate(indices):
        events, label = dataset[int(i)]
        frames = events_to_frames(events, n_time_bins=n_time_bins)
        X.append(frames)
        y.append(label)

        if (count + 1) % 100 == 0:
            print(f"Loaded {count + 1}/{n_samples} samples")

    return np.stack(X), np.array(y)


def main():
    n_time_bins = 90
    train_size = 60000
    test_size = 10000
    total_reservoir_size = 3600
    n_partitions = 3
    grid_shape = (10, 10, 12)
    tau_v = 16.0
    tau_u = 16.0
    threshold = 20.0
    reservoir_weight = 1.0
    input_density = 0.02
    input_weight = 1.0
    lambda_param = 3.0
    inter_partition_density = 0.001
    inter_partition_weight = -1.0

    print("Loading train subset...")
    X_train, y_train = load_subset(
        train=True,
        n_samples=train_size, # TODO 60000
        n_time_bins=n_time_bins,
        seed=SEED
    )

    # TODO and this size
    print("Loading test subset...")
    X_test, y_test = load_subset(
        train=False,
        n_samples=test_size, # TODO 10000
        n_time_bins=n_time_bins,
        seed=SEED
    )

    print("Train labels:", np.bincount(y_train, minlength=10))
    print("Test labels:", np.bincount(y_test, minlength=10))

    input_size = X_train.shape[-1]
    print("Input size:", input_size)

    print("Creating TEPRE...")
    tepre = TEPRE(
        input_size=input_size,
        total_reservoir_size=total_reservoir_size,
        n_partitions=n_partitions,
        grid_shape=grid_shape,
        tau_v=tau_v,
        tau_u=tau_u,
        threshold=threshold,
        reservoir_weight=reservoir_weight,
        input_density=input_density,
        input_weight=input_weight,
        lambda_param=lambda_param,
        inter_partition_density=inter_partition_density,
        inter_partition_weight=inter_partition_weight,
        seed=SEED,
    )

    print("Transforming train data...")
    Z_train = tepre.transform(X_train)

    print("Transforming test data...")
    Z_test = tepre.transform(X_test)

    print("Feature shape train:", Z_train.shape)
    print("Feature shape test:", Z_test.shape)

    print("Training linear classifier...")
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=1000,
            n_jobs=-1,
            solver="lbfgs",
            random_state=SEED
        )
    )
    clf.fit(Z_train, y_train)

    preds = clf.predict(Z_test)
    acc = accuracy_score(y_test, preds)

    print(f"N-MNIST TEPRE accuracy: {acc * 100:.2f}%")

    results = {
        "experiment": "nmnist-tepre",
        "accuracy": float(acc),
        "train_samples": int(len(y_train)),
        "test_samples": int(len(y_test)),
        "seed": SEED,
        "timestamp": datetime.now().isoformat(),
        "model": {
            "total_reservoir_size": total_reservoir_size,
            "n_partitions": n_partitions,
            "tau_v": tau_v,
            "tau_u": tau_u,
            "threshold": threshold,
            "input_density": input_density,
            "input_weight": input_weight,
            "lambda_param": lambda_param,
        }
    }

    save_results(results, "nmnist_tepre.json")
    np.savez(
        PROJECT_ROOT / "results" / "nmnist_tepre_predictions.npz",
        y_true=y_test,
        y_pred=preds,
    )


if __name__ == "__main__":
    main()