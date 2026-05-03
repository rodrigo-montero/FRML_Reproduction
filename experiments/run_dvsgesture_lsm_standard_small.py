import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import numpy as np
import tonic
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from src.datasets.event_utils import dvs_gesture_events_to_frames
from src.models.lsm_paper import PaperLSM


DATA_PATH = PROJECT_ROOT / "data" / "raw"


def load_subset(
    train=True,
    n_samples=100,
    time_window=20000,
    max_time_bins=80,
    seed=42,
):
    dataset = tonic.datasets.DVSGesture(
        save_to=str(DATA_PATH),
        train=train
    )

    rng = np.random.default_rng(seed)
    indices = rng.choice(len(dataset), size=min(n_samples, len(dataset)), replace=False)

    X = []
    y = []

    for count, i in enumerate(indices):
        events, label = dataset[int(i)]

        frames = dvs_gesture_events_to_frames(
            events,
            time_window=time_window,
            max_time_bins=max_time_bins,
        )

        X.append(frames)
        y.append(label)

        if (count + 1) % 10 == 0:
            print(f"Loaded {count + 1}/{len(indices)} samples")

    return np.stack(X), np.array(y)


def main():
    time_window = 20000
    max_time_bins = 80

    print("Loading DVSGesture train subset...")
    X_train, y_train = load_subset(
        train=True,
        n_samples=300,
        time_window=time_window,
        max_time_bins=max_time_bins,
        seed=42
    )

    print("Loading DVSGesture test subset...")
    X_test, y_test = load_subset(
        train=False,
        n_samples=100,
        time_window=time_window,
        max_time_bins=max_time_bins,
        seed=123
    )

    print("Train labels:", np.bincount(y_train, minlength=11))
    print("Test labels:", np.bincount(y_test, minlength=11))
    print("Input shape:", X_train.shape)

    input_size = X_train.shape[-1]

    print("Creating 4000-neuron standard LSM...")

    lsm = PaperLSM(
        input_size=input_size,

        # Paper DVSGesture setting
        reservoir_size=4000,
        grid_shape=(20, 20, 10),
        tau_v=5.0,
        tau_u=10.0,
        threshold=20.0,
        reservoir_weight=1.0,

        # Assumed values
        input_density=0.02,
        input_weight=1.0,
        lambda_param=3.0,
        seed=42,
    )

    print("Transforming train data...")
    Z_train = lsm.transform(X_train)

    print("Transforming test data...")
    Z_test = lsm.transform(X_test)

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

    preds = clf.predict(Z_test)
    acc = accuracy_score(y_test, preds)

    print(f"DVSGesture small standard LSM accuracy: {acc * 100:.2f}%")


if __name__ == "__main__":
    main()