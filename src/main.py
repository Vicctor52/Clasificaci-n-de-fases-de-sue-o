from config import ARTIFACT_THRESHOLD, CHANNELS_TO_DROP
from utils import process_patient


def main():
    """Load and preprocess each patient's recording.

    The ten participants are processed in order and their features are stored
    together with both annotators' labels for the following stages.
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


if __name__ == "__main__":
    main()
