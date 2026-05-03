import tonic
from torch.utils.data import DataLoader

def get_nmnist(batch_size=32):
    transform = None  # add later if needed

    train_set = tonic.datasets.NMNIST(
        save_to="../../data/raw",
        train=True,
        transform=transform
    )

    test_set = tonic.datasets.NMNIST(
        save_to="../../data/raw",
        train=False,
        transform=transform
    )

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size)

    return train_loader, test_loader