import torchvision
import torch


def load_dataset(path, trans):
    train_dataset = torchvision.datasets.ImageFolder(
        root=path,
        transform=trans
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=64,
        num_workers=0,
        shuffle=True
    )
    return train_loader