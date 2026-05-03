from pathlib import Path
import tonic

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "raw"

datasets = [
    ("N-MNIST train", lambda: tonic.datasets.NMNIST(save_to=str(DATA_PATH), train=True)),
    ("N-MNIST test", lambda: tonic.datasets.NMNIST(save_to=str(DATA_PATH), train=False)),
    ("SHD train", lambda: tonic.datasets.SHD(save_to=str(DATA_PATH), train=True)),
    ("SHD test", lambda: tonic.datasets.SHD(save_to=str(DATA_PATH), train=False)),
    ("DVSGesture train", lambda: tonic.datasets.DVSGesture(save_to=str(DATA_PATH), train=True)),
    ("DVSGesture test", lambda: tonic.datasets.DVSGesture(save_to=str(DATA_PATH), train=False)),
]

for name, load_fn in datasets:
    try:
        print(f"Loading/downloading {name}...")
        load_fn()
        print(f"{name} ready.")
    except Exception as e:
        print(f"Failed to download {name}: {e}")

print("Done.")