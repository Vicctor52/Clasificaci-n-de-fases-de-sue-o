from config import ARTIFACT_THRESHOLD, CHANNELS_TO_DROP, train_patients
from model_utils import data_preparation, xgb_definition, xgb_fitting
from utils import process_patient


def main():
    """Preprocess the recordings and train the multiclass model.

    Patient recordings are preprocessed first, then split by patient ID into
    training and test sets. The four-class model is fitted on the consensus
    labels and used to predict the test epochs.
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


if __name__ == "__main__":
    main()
