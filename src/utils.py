from pathlib import Path

import mne
import pandas as pd
import yasa


def get_edf_patient(patient_id, base_path=None):
    """Get a patient's EDF path and load both annotators' labels.

    Args:
        patient_id: Patient identifier used in filenames and the data folder.
        base_path: Directory containing patient subfolders. Defaults to the
            project's data directory.

    Returns:
        A tuple containing the EDF path and both annotators' label arrays.
    """
    if base_path is None:
        base_path = Path(__file__).resolve().parent.parent / "data"
    else:
        base_path = Path(base_path)

    patient_folder = base_path / str(patient_id)
    edf_path = patient_folder / f"{patient_id}.edf"

    labels = []
    for annotator_id in [1, 2]:
        label_path = patient_folder / f"{patient_id}_{annotator_id}.txt"
        label_series = pd.read_csv(label_path, header=None).squeeze("columns")
        label_series[label_series == 5] = 4
        labels.append(label_series[:-30].values)

    return edf_path, labels


def process_patient(patient_id, channels_to_drop, epoch_length=30, base_path=None):
    """Extract sleep features and labels for a patient.

    Args:
        patient_id: Patient identifier.
        channels_to_drop: Channel names to remove from the EDF recording.
        epoch_length: Epoch duration in seconds.
        base_path: Directory containing patient subfolders. Defaults to the
            project's data directory.

    Returns:
        A tuple containing the feature table and the two filtered label arrays.
    """
    edf_path, labels = get_edf_patient(patient_id, base_path=base_path)

    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    raw.drop_channels([channel for channel in channels_to_drop if channel in raw.ch_names])

    raw.resample(100)
    raw.filter(0.3, 49, verbose=False)

    sleep_staging = yasa.SleepStaging(
        raw,
        eeg_name='C4-A1',
        eog_name='LOC-A2',
        emg_name='X1',
    )
    features = sleep_staging.get_features()[:-30]

    return features, labels[0], labels[1]
