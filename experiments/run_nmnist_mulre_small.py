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

from src.datasets.event_utils import events_to_frames
from src.models.mulre import MuLRE


DATA_PATH = PROJECT_ROOT / "data" / "raw"


def load_subset(train=True, n_samples=1000, n_time_bins=90, seed=42):
    dataset = tonic.datasets.NMNIST(
        save_to=str(DATA_PATH),
        train=train
    )

    rng = np.random.default_rng(seed)
    indices = rng.choice(len(dataset), size=min(n_samples, len(dataset)), replace=False)

    X = []
    y = []

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
    n_time_bins = 90

    print("Loading N-MNIST train subset...")
    X_train, y_train = load_subset(
        train=True,
        n_samples=1000,
        n_time_bins=n_time_bins,
        seed=42
    )

    print("Loading N-MNIST test subset...")
    X_test, y_test = load_subset(
        train=False,
        n_samples=300,
        n_time_bins=n_time_bins,
        seed=123
    )

    print("Train labels:", np.bincount(y_train, minlength=10))
    print("Test labels:", np.bincount(y_test, minlength=10))
    print("Input shape:", X_train.shape)

    print("Creating MuLRE...")

    mulre = MuLRE(
        input_shape=(34, 34, 2),

        # Small debug setting:
        # 1500 total = 3 reservoirs of 500 each.
        # grid_shape must multiply to 500.
        total_reservoir_size=1500,
        n_reservoirs=3,
        grid_shape=(10, 10, 5),

        # Paper values for 3-reservoir MuLRE
        d_values=[0, 4, 6],

        # Paper says receptive-field window is 5 or 6.
        receptive_field_size=6,

        tau_v=16.0,
        tau_u=16.0,
        threshold=20.0,
        reservoir_weight=1.0,

        # Assumed values
        input_density=0.02,
        input_weight=1.0,
        lambda_param=3.0,
        seed=42,
    )

    print("Transforming train data...")
    Z_train = mulre.transform(X_train)

    print("Transforming test data...")
    Z_test = mulre.transform(X_test)

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
            solver="lbfgs"
        )
    )

    clf.fit(Z_train, y_train)

    clf.fit(Z_train, y_train)

    preds = clf.predict(Z_test)
    acc = accuracy_score(y_test, preds)

    print(f"N-MNIST small MuLRE accuracy: {acc * 100:.2f}%")


if __name__ == "__main__":
    main()