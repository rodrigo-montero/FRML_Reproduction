from pathlib import Path
import tonic

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "raw"

dataset = tonic.datasets.NMNIST(
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