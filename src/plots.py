import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc, confusion_matrix, roc_curve, ConfusionMatrixDisplay

from config import COMBINED_CLASS_LABELS, COMBINED_CLASS_NAMES


def plot_roc(y_true, probabilities, labels, class_names=None, title=None, show=True):
    """Plot a binary ROC curve or one-vs-rest curves for multiple classes.

    Args:
        y_true: True class labels.
        probabilities: Positive-class scores for binary classification, or
            class probabilities with columns ordered as ``labels``.
        labels: Class labels ordered to match the probability columns.
        class_names: Optional display names in the same order as ``labels``.
        title: Optional plot title.
        show: Whether to display the plot interactively.

    Returns:
        A dictionary containing the AUC for each plotted class. For
        multiclass data, it also contains the macro-average AUC.
    """
    y_true = np.asarray(y_true)
    probabilities = np.asarray(probabilities)
    labels = np.asarray(labels)
    if class_names is None:
        class_names = [str(label) for label in labels]

    class_aucs = {}
    plt.figure(figsize=(10, 8))
    class_indices = range(labels.size) if labels.size > 2 else [1]
    for class_index in class_indices:
        binary_labels = (y_true == labels[class_index]).astype(int)
        if np.unique(binary_labels).size < 2:
            continue
        scores = probabilities if probabilities.ndim == 1 else probabilities[:, class_index]
        fpr, tpr, _ = roc_curve(binary_labels, scores)
        class_auc = auc(fpr, tpr)
        class_aucs[class_names[class_index]] = class_auc
        plt.plot(
            fpr,
            tpr,
            lw=2,
            label=f"Curva ROC de {class_names[class_index]} (AUC = {class_auc:.4f})",
        )

    if labels.size > 2 and class_aucs:
        class_aucs["macro_auc"] = float(np.mean(list(class_aucs.values())))

    plt.plot(
        [0, 1],
        [0, 1],
        color="navy",
        lw=2,
        linestyle="--",
        label="Clasificador aleatorio",
    )
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Tasa de falsos positivos", fontsize=12)
    plt.ylabel("Tasa de verdaderos positivos", fontsize=12)
    plt.title(title or "Curva ROC")
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    if show:
        plt.show()

    return class_aucs


def plot_confusion_matrix(
    y_true, y_pred, labels, class_names=None, title=None, show=True
):
    """Plot and return a confusion matrix.

    Args:
        y_true: True class labels.
        y_pred: Predicted class labels.
        labels: Class labels in the desired matrix order.
        class_names: Optional display names in the same order as ``labels``.
        title: Optional plot title.
        show: Whether to display the plot interactively.

    Returns:
        The confusion matrix.
    """
    if class_names is None:
        class_names = [str(label) for label in labels]

    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    display = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=class_names)
    display.plot(cmap="Blues", values_format="d")
    display.ax_.set_xlabel("Etiqueta predicha")
    display.ax_.set_ylabel("Etiqueta real")
    plt.title(title or "Matriz de confusión")
    plt.tight_layout()
    if show:
        plt.show()
    return matrix
