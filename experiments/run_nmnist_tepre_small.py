import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import numpy as np
import tonic
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from src.datasets.event_utils import events_to_frames
from src.models.tepre import TEPRE


DATA_PATH = PROJECT_ROOT / "data" / "raw"


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
    n_partitions = 3

    # TODO change the data sizes to maximum? will take a lot FYI
    print("Loading train subset...")
    X_train, y_train = load_subset(
        train=True,
        n_samples=1000,
        n_time_bins=n_time_bins,
        seed=42
    )

    # TODO and this size
    print("Loading test subset...")
    X_test, y_test = load_subset(
        train=False,
        n_samples=300,
        n_time_bins=n_time_bins,
        seed=123
    )

    print("Train labels:", np.bincount(y_train, minlength=10))
    print("Test labels:", np.bincount(y_test, minlength=10))

    input_size = X_train.shape[-1]
    print("Input size:", input_size)

    print("Creating TEPRE...")
    tepre = TEPRE(
        input_size=input_size,
        total_reservoir_size=3600,
        n_partitions=3,
        grid_shape=(10, 10, 12),
        tau_v=16.0,
        tau_u=16.0,
        threshold=20.0,
        reservoir_weight=1.0,
        lambda_param=3.0,
        input_density=0.02,
        input_weight=1.0,
        inter_partition_density=0.001,
        inter_partition_weight=-1.0,
        seed=42,
    )

    print("Transforming train data...")
    Z_train = tepre.transform(X_train)

    print("Transforming test data...")
    Z_test = tepre.transform(X_test)

    print("Feature shape train:", Z_train.shape)
    print("Feature shape test:", Z_test.shape)

    print("Training linear classifier...")
    clf = LogisticRegression(
        max_iter=1000,
        n_jobs=-1,
        solver="lbfgs"
    )
    clf.fit(Z_train, y_train)

    preds = clf.predict(Z_test)
    acc = accuracy_score(y_test, preds)

    print(f"N-MNIST small TEPRE accuracy: {acc * 100:.2f}%")


if __name__ == "__main__":
    main()