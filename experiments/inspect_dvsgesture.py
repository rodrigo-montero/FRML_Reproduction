import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import tonic

DATA_PATH = PROJECT_ROOT / "data" / "raw"


def main():
    print("Loading DVSGesture train dataset...")

    dataset = tonic.datasets.DVSGesture(
        save_to=str(DATA_PATH),
        train=True
    )

    print("Dataset loaded.")
    print("Number of training samples:", len(dataset))

    events, label = dataset[0]

    print("\nFirst sample:")
    print("Label:", label)
    print("Events type:", type(events))
    print("Events shape:", events.shape)
    print("Events dtype:", events.dtype)

    print("\nFirst 10 events:")
    print(events[:10])

    print("\nEvent fields:")
    print(events.dtype.names)

    # Extra useful diagnostics
    if len(events) > 0:
        print("\nTime range:")
        print("min t:", events["t"].min())
        print("max t:", events["t"].max())

        if "x" in events.dtype.names:
            print("\nX range:", events["x"].min(), events["x"].max())

        if "y" in events.dtype.names:
            print("Y range:", events["y"].min(), events["y"].max())

        if "p" in events.dtype.names:
            print("Polarity values:", set(events["p"]))


if __name__ == "__main__":
    main()