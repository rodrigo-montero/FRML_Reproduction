import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import numpy as np
import tonic
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from src.datasets.event_utils import shd_events_to_bins
from src.models.tepre import TEPRE


DATA_PATH = PROJECT_ROOT / "data" / "raw"


def load_subset(train=True, n_samples=1000, n_time_bins=1000, input_size=700, seed=42):
    dataset = tonic.datasets.SHD(
        save_to=str(DATA_PATH),
        train=train
    )

    rng = np.random.default_rng(seed)
    indices = rng.choice(len(dataset), size=min(n_samples, len(dataset)), replace=False)

    X = []
    y = []

    for count, i in enumerate(indices):
        events, label = dataset[int(i)]

        frames = shd_events_to_bins(
            events,
            n_time_bins=n_time_bins,
            input_size=input_size
        )

        X.append(frames)
        y.append(label)

        if (count + 1) % 100 == 0:
            print(f"Loaded {count + 1}/{len(indices)} samples")

    return np.stack(X), np.array(y)


def main():
    n_time_bins = 1000
    input_size = 700

    print("Loading SHD train subset...")
    X_train, y_train = load_subset(
        train=True,
        n_samples=500,
        n_time_bins=n_time_bins,
        input_size=input_size,
        seed=42
    )

    print("Loading SHD test subset...")
    X_test, y_test = load_subset(
        train=False,
        n_samples=200,
        n_time_bins=n_time_bins,
        input_size=input_size,
        seed=123
    )

    print("Train labels:", np.bincount(y_train))
    print("Test labels:", np.bincount(y_test))
    print("Input shape:", X_train.shape)

    print("Creating SHD TEPRE...")

    tepre = TEPRE(
        input_size=input_size,
        total_reservoir_size=3000,
        n_partitions=6,

        # Paper gives total grid Nx=Ny=10, Nz=30.
        # With 6 partitions, each partition is 10x10x5 = 500 neurons.
        grid_shape=(10, 10, 5),

        # Paper SHD values
        tau_v=40.0,
        tau_u=20.0,
        threshold=20.0,
        reservoir_weight=1.0,

        # Assumed values: paper does not fully specify these
        input_density=0.02,
        input_weight=1.0,
        lambda_param=3.0,
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

    print("Mean train feature activity:", Z_train.mean())
    print("Max train feature activity:", Z_train.max())
    print("Nonzero train feature fraction:", (Z_train > 0).mean())

    print("Training linear classifier...")
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=3000,
            n_jobs=-1,
            solver="lbfgs",
            C=1.0
        )
    )

    clf.fit(Z_train, y_train)

    clf.fit(Z_train, y_train)

    preds = clf.predict(Z_test)
    acc = accuracy_score(y_test, preds)

    print(f"SHD small TEPRE accuracy: {acc * 100:.2f}%")


if __name__ == "__main__":
    main()