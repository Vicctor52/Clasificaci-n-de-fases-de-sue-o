from config import (
    ARTIFACT_THRESHOLD,
    CHANNELS_TO_DROP,
    CONFIDENCE_THRESHOLD,
    train_patients,
)
from model_utils import (
    binary_data_preparation,
    binary_xgb_definition,
    binary_xgb_evaluation,
    binary_xgb_fitting,
    data_preparation,
    get_confidence_mask,
    model_combination,
    xgb_definition,
    xgb_evaluation,
    xgb_fitting,
)
from utils import process_patient


def main():
    """Preprocess the recordings and train both classification models.

    Patient recordings are preprocessed first, then split by patient ID into
    training and test sets. The multiclass and binary models are fitted on the
    consensus labels and evaluated on the test epochs.
    """
    data_list = []
    for patient_id in range(1, 11):
        print(f"Processing patient {patient_id}...")
        features, labels_1, labels_2 = process_patient(
            patient_id,
            CHANNELS_TO_DROP,
            remove_artifacts=True,
            artifact_threshold=ARTIFACT_THRESHOLD,
            show_plot=False,
        )
        data_list.append(
            {
                "id": patient_id,
                "X": features,
                "y1": labels_1,
                "y2": labels_2,
            }
        )

    print(f"\nProcessing complete. Patients loaded: {len(data_list)}")

    X_train, X_test, y_train, y_test, y_train_original, y_test_original = data_preparation(
        data_list, train_patients=train_patients
    )
    print(f"Training samples: {len(X_train)} | Test samples: {len(X_test)}")

    grid_search = xgb_definition()
    calibrated_model = xgb_fitting(X_train, y_train, grid_search)
    multiclass_predictions = calibrated_model.predict(X_test)
    print(f"Predicted multiclass labels: {len(multiclass_predictions)}")

    xgb_evaluation(calibrated_model, X_test, y_test)

    # Train a second model to distinguish N1 from N2 within the light-sleep class.
    X_train_bin, y_train_bin, X_test_bin, y_test_bin = binary_data_preparation(
        X_train, X_test, y_train_original, y_test_original
    )
    grid_search_bin = binary_xgb_definition()
    calibrated_model_bin = binary_xgb_fitting(
        grid_search_bin, X_train_bin, y_train_bin
    )
    binary_probabilities = calibrated_model_bin.predict_proba(X_test_bin)
    binary_threshold = 0.5

    binary_predictions = (binary_probabilities[:, 1] > binary_threshold).astype(int)

    binary_xgb_evaluation(
        calibrated_model_bin,
        X_test_bin,
        y_test_bin,
        threshold=binary_threshold,
    )

    combined_predictions, combined_probabilities, _ = model_combination(
        X_test,
        calibrated_model,
        calibrated_model_bin,
    )
    confidence_mask = get_confidence_mask(
        combined_probabilities, threshold=CONFIDENCE_THRESHOLD
    )
    print(
        "Predictions retained by confidence threshold: "
        f"{confidence_mask.sum()} / {len(confidence_mask)}"
    )


if __name__ == "__main__":
    main()
