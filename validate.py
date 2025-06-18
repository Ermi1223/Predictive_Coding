import torch
from utils.visualization import plot_confusion_matrix, show_misclassified_images
from utils.metrics import print_classification_report

def validate(pcn, classifier, test_loader, device, show_confusion=True, save_path=None, print_report=True):
    pcn.eval()
    classifier.eval()
    correct = 0
    total = 0

    all_preds = []
    all_labels = []
    all_images = []

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            pcn.reset_states()
            for _ in range(5):
                _, _, _, h2 = pcn(x)
            logits = classifier(h2)
            preds = logits.argmax(dim=1)

            all_preds.append(preds.cpu())
            all_labels.append(y.cpu())
            all_images.append(x.cpu())

            correct += (preds == y).sum().item()
            total += y.size(0)

    acc = correct / total * 100
    print(f"[Test] Accuracy: {acc:.2f}%")

    y_true = torch.cat(all_labels)
    y_pred = torch.cat(all_preds)
    raw_images = torch.cat(all_images)

    if print_report:
        report_path = None
        if save_path:
            report_path = save_path.replace(".png", "_report.txt")
        print_classification_report(y_true, y_pred, save_path=report_path)

    if show_confusion:
        plot_confusion_matrix(y_true, y_pred, save_path)

    show_misclassified_images(raw_images, y_true, y_pred)

    return acc
