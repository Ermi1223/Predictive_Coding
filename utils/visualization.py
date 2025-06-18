import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import os

def plot_confusion_matrix(y_true, y_pred, save_path=None):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(range(10)))
    disp.plot(cmap='Blues', values_format='d')
    plt.title("MNIST Confusion Matrix")
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"[Saved] Confusion matrix to: {save_path}")
    plt.show()

def show_misclassified_images(images, labels, preds, max_images=10):
    wrong = preds != labels
    wrong_images = images[wrong]
    wrong_preds = preds[wrong]
    wrong_labels = labels[wrong]

    print(f"[Info] Total misclassified: {len(wrong_images)}")
    if len(wrong_images) == 0:
        print("✅ No misclassifications!")
        return

    max_images = min(max_images, len(wrong_images))
    fig, axes = plt.subplots(1, max_images, figsize=(1.5 * max_images, 2))
    for i in range(max_images):
        ax = axes[i]
        ax.imshow(wrong_images[i].squeeze(), cmap='gray')
        ax.set_title(f"P:{wrong_preds[i]} / T:{wrong_labels[i]}")
        ax.axis('off')
    plt.tight_layout()
    plt.suptitle("Misclassified MNIST Digits", y=1.05, fontsize=16)
    plt.show()
