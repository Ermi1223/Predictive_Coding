from sklearn.metrics import classification_report

def print_classification_report(y_true, y_pred, save_path=None):
    report = classification_report(y_true, y_pred, digits=4)
    print("\n[Per-Class Metrics]\n" + report)
    if save_path:
        with open(save_path, "w") as f:
            f.write(report)
