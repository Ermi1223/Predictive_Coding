from sklearn.metrics import classification_report
from typing import Optional, Union
import numpy as np
import torch

def print_classification_report(
    y_true: Union[np.ndarray, torch.Tensor],
    y_pred: Union[np.ndarray, torch.Tensor],
    save_path: Optional[str] = None
) -> None:
    # Convert to numpy if input is torch.Tensor
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()

    report = classification_report(y_true, y_pred, digits=4)
    print("\n[Per-Class Metrics]\n" + report)

    if save_path:
        with open(save_path, "w") as f:
            f.write(report)
