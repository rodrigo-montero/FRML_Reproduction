import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

import random
import json
from datetime import datetime
import numpy as np
import tonic
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score

from src.datasets.event_utils import dvs_gesture_events_to_frames
from src.models.lsm_paper import ReceptiveFieldLSM


DATA_PATH = PROJECT_ROOT / "data" / "raw"

SEEDS = [21, 42, 67]

#random.seed(SEED)
#np.random.seed(SEED)

def save_results(results, filename):
    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(exist_ok=True)

    path = results_dir / filename

    with open(path, "w") as f:
        json.dump(results, f, indent=4)

    print(f"Saved results to {path}")


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
    train_size = 1077 # Max is 1077
    test_size = 264 # Max is 264
    input_shape = (64, 64, 2)
    reservoir_size = 4000
    grid_shape = (20, 20, 10)
    tau_v = 5.0
    tau_u = 10.0
    threshold = 20.0
    reservoir_weight = 1.0
    receptive_field_size = 6 # Paper says window size 5 or 6

    # Assumed values
    input_density = 0.02
    input_weight = 1.0
    lambda_param = 3.0
    # And seed

    all_results = []

    for seed in SEEDS:
        print(f"\n{'=' * 50}")
        print(f"Running seed {seed}")
        print(f"{'=' * 50}")
        print("Creating receptive-field LSM...")

        print("Loading DVSGesture train subset...")
        X_train, y_train = load_subset(
            train=True,
            n_samples=train_size,
            time_window=time_window,
            max_time_bins=max_time_bins,
            seed=seed
        )

        print("Loading DVSGesture test subset...")
        X_test, y_test = load_subset(
            train=False,
            n_samples=test_size,
            time_window=time_window,
            max_time_bins=max_time_bins,
            seed=seed
        )

        print("Train labels:", np.bincount(y_train, minlength=11))
        print("Test labels:", np.bincount(y_test, minlength=11))
        print("Input shape:", X_train.shape)

        lsm = ReceptiveFieldLSM(
            input_shape=input_shape,
            reservoir_size=reservoir_size,
            grid_shape=grid_shape,
            tau_v=tau_v,
            tau_u=tau_u,
            threshold=threshold,
            reservoir_weight=reservoir_weight,
            receptive_field_size=receptive_field_size,
            input_density=input_density,
            input_weight=input_weight,
            lambda_param=lambda_param,
            seed=seed,
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
        clf = SGDClassifier(max_iter=10000, tol=1e-6, random_state=seed)

        clf.fit(Z_train, y_train)

        preds = clf.predict(Z_test)
        acc = accuracy_score(y_test, preds)

        print(f"DVSGesture receptive-field LSM seed: {seed}, accuracy: {acc * 100:.2f}%")

        results = {
            "experiment": "dvsgesture-lsm-receptive",
            "accuracy": float(acc),
            "train_samples": int(len(y_train)),
            "test_samples": int(len(y_test)),
            "seed": seed,
            "timestamp": datetime.now().isoformat(),
            "model": {
                "total_reservoir_size": reservoir_size,
                #"n_partitions": 3,
                "tau_v": tau_v,
                "tau_u": tau_u,
                "threshold": threshold,
                "input_density": input_density,
                "input_weight": input_weight,
                "lambda_param": lambda_param,
            }
        }
        all_results.append(results)

        save_results(results, f"dvsgesture_lsm_receptive_seed{seed}.json")
        np.savez(
            PROJECT_ROOT / "results" / f"dvsgesture_lsm_receptive_predictions_seed{seed}.npz",
            y_true=y_test,
            y_pred=preds,
        )
    # Summary across seeds
    accuracies = [r["accuracy"] for r in all_results]
    summary = {
        "experiment": "dvsgesture-lsm-receptive",
        "seeds": SEEDS,
        "accuracies": accuracies,
        "mean_accuracy": float(np.mean(accuracies)),
        "std_accuracy": float(np.std(accuracies)),
        "timestamp": datetime.now().isoformat(),
    }
    save_results(summary, "dvsgesture_lsm_receptive_summary.json")
    print(f"\nMean accuracy: {np.mean(accuracies) * 100:.2f}% ± {np.std(accuracies) * 100:.2f}%")


if __name__ == "__main__":
    main()