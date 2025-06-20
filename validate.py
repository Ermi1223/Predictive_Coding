import torch
from utils.visualization import plot_confusion_matrix, show_misclassified_images
from utils.metrics import print_classification_report

def validate(model, test_loader, device, recurrent_steps=5,
             show_confusion=True, save_path=None, print_report=True):
    model.eval()

    correct = 0
    total = 0

    all_preds = []
    all_labels = []
    all_images = []

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)

            # If your model needs resetting hidden states between batches:
            if hasattr(model, 'reset_states'):
                model.reset_states()

            # Forward pass with recurrent steps
            logits = model(x, steps=recurrent_steps)
            preds = logits.argmax(dim=1)

            all_preds.append(preds.cpu())
            all_labels.append(y.cpu())
            all_images.append(x.cpu())

            correct += (preds == y).sum().item()
            total += y.size(0)

    accuracy = correct / total * 100
    print(f"[Test] Accuracy: {accuracy:.2f}%")

    y_true = torch.cat(all_labels)
    y_pred = torch.cat(all_preds)
    raw_images = torch.cat(all_images)

    if print_report:
        report_path = save_path.replace(".png", "_report.txt") if save_path else None
        print_classification_report(y_true, y_pred, save_path=report_path)

    if show_confusion:
        plot_confusion_matrix(y_true, y_pred, save_path)

    show_misclassified_images(raw_images, y_true, y_pred)

    return accuracy
