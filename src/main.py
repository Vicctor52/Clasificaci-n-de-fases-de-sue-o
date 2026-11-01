from config import (
    ARTIFACT_THRESHOLD,
    CHANNELS_TO_DROP,
    CONFIDENCE_THRESHOLD,
    train_patients,
)
from utils import process_patient
from model_utils import (
    binary_data_preparation,
    binary_xgb_definition,
    binary_xgb_fitting,
    data_preparation,
    get_confidence_mask,
    model_combination,
    xgb_definition,
    xgb_fitting,
)
from report import generate_report


def main():
    """Run the sleep-stage classification pipeline and create its report.

    Patient recordings are preprocessed first, then split by patient ID into
    training and test sets. The multiclass and binary models are fitted
    separately, their predictions are combined, and the final HTML report
    summarizes model performance with and without confidence filtering.
    """
    # Preprocess each patient and collect features with both annotators' labels.
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

    # Keep only epochs where annotators agree and split patients into train/test.
    X_train, X_test, y_train, y_test, y_train_original, y_test_original = data_preparation(
        data_list, train_patients=train_patients
    )

    # Fit the four-class model and retain its test predictions and probabilities.
    grid_search = xgb_definition()
    calibrated_model = xgb_fitting(X_train, y_train, grid_search)
    multiclass_predictions = calibrated_model.predict(X_test)
    multiclass_probabilities = calibrated_model.predict_proba(X_test)

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

    # Combine both models into five sleep stages and mark confident predictions.
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

    # Generate a report using test results and the patient-level split details.
    generate_report(
        y_true_multiclass=y_test,
        y_pred_multiclass=multiclass_predictions,
        probabilities_multiclass=multiclass_probabilities,
        y_true_binary=y_test_bin,
        y_pred_binary=binary_predictions,
        probabilities_binary=binary_probabilities[:, 1],
        y_true_combined=y_test_original,
        y_pred_combined=combined_predictions,
        probabilities_combined=combined_probabilities,
        confidence_mask=confidence_mask,
        confidence_threshold=CONFIDENCE_THRESHOLD,
        artifact_threshold=ARTIFACT_THRESHOLD,
        train_patient_ids=[
            patient["id"]
            for patient in data_list
            if patient["id"] <= train_patients
        ],
        test_patient_ids=[
            patient["id"]
            for patient in data_list
            if patient["id"] > train_patients
        ],
    )


if __name__ == "__main__":
    main()
