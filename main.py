import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from train import train_epoch
from validate import validate
from models.predictive_net import PredictiveCodingNet
from utils.visualization import plot_confusion_matrix

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.ToTensor()
    train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

    model = PredictiveCodingNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()

    epochs = 10
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)

    for epoch in range(1, epochs + 1):
        train_epoch(model, train_loader, optimizer, criterion, device, epoch, recurrent_steps=5)

        accuracy = validate(model, test_loader, device, recurrent_steps=5, show_confusion=False)

        print(f"Epoch {epoch} completed. Validation Accuracy: {accuracy:.4f}")

        # Optionally save model checkpoint
        torch.save(model.state_dict(), os.path.join(output_dir, f"pcn_epoch_{epoch}.pth"))

if __name__ == "__main__":
    main()
