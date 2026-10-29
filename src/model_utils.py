import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from config import (
    BINARY_CLASS_LABELS,
    BINARY_CLASS_NAMES,
    MULTICLASS_CLASS_LABELS,
    MULTICLASS_CLASS_NAMES,
)
from plots import plot_confusion_matrix, plot_roc

#Multiclass model

def data_preparation(data_list, train_patients=8):
    """Prepare consensus-labeled training and test sets for both models.

    Args:
        data_list: Patient records containing IDs, features, and two label arrays.

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


#Binary model
def binary_data_preparation(X_train,X_test,y_train_original, y_test_original):
    """Select N1 and N2 rows and remap their labels for binary classification.

    Args:
        X_train: Training feature table.
        X_test: Test feature table.
        y_train_original: Original training labels, with N1 and N2 encoded as 1 and 2.
        y_test_original: Original test labels, with N1 and N2 encoded as 1 and 2.

    Returns:
        Filtered training features and labels, followed by filtered test features
        and labels. N1 is encoded as 0 and N2 as 1.
    """
    train_mask = np.isin(y_train_original, [1, 2])
    test_mask = np.isin(y_test_original, [1, 2])

    X_train_bin = X_train.loc[train_mask]
    y_train_bin = y_train_original[train_mask] - 1

    X_test_bin = X_test.loc[test_mask]
    y_test_bin = y_test_original[test_mask] - 1

    return X_train_bin, y_train_bin, X_test_bin, y_test_bin

def binary_xgb_definition():
    """Create the binary XGBoost hyperparameter search.

    Args:
        None.

    Returns:
        GridSearchCV configured for binary classification.
    """
    xgb_model_bin = xgb.XGBClassifier(
        objective='binary:logistic',
        tree_method='hist',
        random_state=42,
        n_jobs=-1
    )

    param_grid_bin = {
        'n_estimators': [100, 200],
        'max_depth': [3, 6],
        'learning_rate': [0.1],
        'subsample': [0.8]
    }

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    grid_search_bin = GridSearchCV(
        estimator=xgb_model_bin,
        param_grid=param_grid_bin,
        scoring='accuracy',
        cv=cv,
        verbose=1
    )
    return grid_search_bin


def binary_xgb_fitting(grid_search_bin, X_train_bin, y_train_bin):
    """Fit and calibrate the binary model using a parameter search.

    Args:
        grid_search_bin: Configured GridSearchCV instance.
        X_train_bin: Feature table containing N1 and N2 training rows.
        y_train_bin: Binary training labels, with N1 as 0 and N2 as 1.

    Returns:
        The fitted probability-calibrated binary classifier.
    """
    grid_search_bin.fit(X_train_bin, y_train_bin)

    best_model_bin = grid_search_bin.best_estimator_
    print(f"Best parameters: {grid_search_bin.best_params_}")

    calibrated_model_bin = CalibratedClassifierCV(best_model_bin, method='isotonic', cv=5)

    calibrated_model_bin.fit(X_train_bin, y_train_bin)
    return calibrated_model_bin

def binary_xgb_evaluation(
    calibrated_model_bin,
    X_test_bin,
    y_test_bin,
    threshold=0.5,
    class_names=BINARY_CLASS_NAMES,
    class_labels=BINARY_CLASS_LABELS,
):
    """Evaluate the binary model with a confusion matrix and ROC curve.

    Args:
        calibrated_model_bin: Fitted calibrated binary classifier.
        X_test_bin: Feature table containing N1 and N2 test rows.
        y_test_bin: True binary test labels, with N1 as 0 and N2 as 1.
        threshold: Probability threshold used to predict the positive class.
        class_names: Display names corresponding to the class labels.
        class_labels: Class labels ordered to match probability columns.

    Returns:
        None.
    """
    y_prob = calibrated_model_bin.predict_proba(X_test_bin)

    # Predict the positive class when its probability exceeds the threshold.
    y_pred_threshold = (y_prob[:, 1] > threshold).astype(int)

    plot_confusion_matrix(
        y_test_bin,
        y_pred_threshold,
        labels=class_labels,
        class_names=class_names,
        title="Matriz de confusión - Modelo binario",
    )
    class_aucs = plot_roc(
        y_test_bin,
        y_prob[:, 1],
        labels=class_labels,
        class_names=class_names,
        title="Curva ROC - Modelo binario",
    )
    print(f"\nOverall binary AUC: {class_aucs[class_names[1]]:.4f}")
