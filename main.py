import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import os

from models.predictive_net import PredictiveCodingNet
from train import train_epoch
from validate import validate

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    transform = transforms.ToTensor()

    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(root="./data", train=True, transform=transform, download=True),
        batch_size=64, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(root="./data", train=False, transform=transform, download=True),
        batch_size=256, shuffle=False)

    pcn = PredictiveCodingNet().to(device)
    classifier = nn.Linear(64, 10).to(device)

    optimizer = optim.Adam(list(pcn.parameters()) + list(classifier.parameters()), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    best_acc = 0
    os.makedirs("outputs", exist_ok=True)

    for epoch in range(1, 6):
        train_epoch(pcn, classifier, train_loader, optimizer, loss_fn, device, epoch)
        acc = validate(
            pcn, classifier, test_loader, device,
            show_confusion=True,
            save_path=f"outputs/conf_matrix_epoch_{epoch}.png"
        )
        if acc > best_acc:
            best_acc = acc
            torch.save({
                'pcn': pcn.state_dict(),
                'classifier': classifier.state_dict()
            }, "best_model.pth")
            print(f"✅ Best model saved with accuracy {acc:.2f}%")

if __name__ == "__main__":
    main()
