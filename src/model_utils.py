import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import ConfusionMatrixDisplay, auc, confusion_matrix, roc_curve
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from config import MULTICLASS_CLASS_LABELS, MULTICLASS_CLASS_NAMES


def data_preparation(data_list, train_patients=8):
    """Prepare consensus-labeled training and test sets.

    Args:
        data_list: Patient records containing IDs, features, and two label arrays.
        train_patients: Highest patient ID assigned to the training split.

    Returns:
        Training and test features, remapped multiclass labels, and original labels.
    """
    X_train_list, y_train_list = [], []
    X_test_list, y_test_list = [], []

    for p in data_list:
        # Use agreement between both annotators for training patients.
        if p['id'] <= train_patients:

            # Keep epochs where both annotators agree.
            mask_agreement = (p['y1'] == p['y2'])

            X_train_list.append(p['X'][mask_agreement])
            y_train_list.append(p['y1'][mask_agreement])

        else:

            mask_agreement = (p['y1'] == p['y2'])
            X_test_list.append(p['X'][mask_agreement])
            y_test_list.append(p['y1'][mask_agreement])

    # Concatenate the filtered data.
    X_train = pd.concat(X_train_list, ignore_index=True)
    X_test = pd.concat(X_test_list, ignore_index=True)
    y_train_original = np.concatenate(y_train_list)
    y_test_original = np.concatenate(y_test_list)

    y_train_multiclass = np.where(y_train_original > 1, y_train_original - 1, y_train_original)
    y_test_multiclass = np.where(y_test_original > 1, y_test_original - 1, y_test_original)

    return X_train, X_test, y_train_multiclass, y_test_multiclass, y_train_original, y_test_original


def xgb_definition():
    """Create the multiclass XGBoost hyperparameter search.

    Args:
        None.

    Returns:
        GridSearchCV configured for multiclass classification.
    """
    xgb_model = xgb.XGBClassifier(
        objective='multi:softprob',
        num_class=4,
        tree_method='hist',
        random_state=42,
        n_jobs=-1
    )

    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [3, 6],
        'learning_rate': [0.1],
        'subsample': [0.8]
    }

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    grid_search = GridSearchCV(
        estimator=xgb_model,
        param_grid=param_grid,
        scoring='accuracy',
        cv=cv,
        verbose=1
    )
    return grid_search


def xgb_fitting(X_train, y_train, grid_search):
    """Fit and calibrate the multiclass model using a parameter search.

    Args:
        X_train: Feature table used to train the model.
        y_train: Multiclass labels corresponding to the training rows.
        grid_search: Configured GridSearchCV instance.

    Returns:
        The fitted probability-calibrated classifier.
    """
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    print(f"Best parameters: {grid_search.best_params_}")

    calibrated_model = CalibratedClassifierCV(best_model, method='isotonic', cv=5)

    calibrated_model.fit(X_train, y_train)
    return calibrated_model


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


def xgb_evaluation(
    calibrated_model,
    X_test,
    y_test,
    class_names=MULTICLASS_CLASS_NAMES,
    class_labels=MULTICLASS_CLASS_LABELS,
):
    """Evaluate the multiclass model with a confusion matrix and ROC curves.

    Args:
        calibrated_model: Fitted calibrated multiclass classifier.
        X_test: Feature table used for evaluation.
        y_test: True multiclass labels for the test rows.
        class_names: Display names corresponding to the class labels.
        class_labels: Class labels ordered to match probability columns.

    Returns:
        None.
    """
    y_pred = calibrated_model.predict(X_test)
    y_prob = calibrated_model.predict_proba(X_test)

    plot_confusion_matrix(
        y_test,
        y_pred,
        labels=class_labels,
        class_names=class_names,
        title="Matriz de confusión - Modelo multiclase",
    )
    class_aucs = plot_roc(
        y_test,
        y_prob,
        labels=class_labels,
        class_names=class_names,
        title="Curvas ROC - Modelo multiclase",
    )
    print(f"\nOverall multiclass AUC: {class_aucs['macro_auc']:.4f}")
