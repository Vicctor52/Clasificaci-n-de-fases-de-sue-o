import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import GridSearchCV, StratifiedKFold


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
