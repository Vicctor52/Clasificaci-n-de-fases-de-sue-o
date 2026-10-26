from pathlib import Path
import mne
import yasa
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
from config import ARTIFACT_THRESHOLD


def artifact_graph(threshold, max_peak_to_peak_per_epoch):
    """Plot epoch amplitudes and the artifact-rejection threshold.

    Args:
        threshold: Peak-to-peak amplitude threshold.
        max_peak_to_peak_per_epoch: Maximum channel amplitude for each epoch.

    Returns:
        None.
    """
    print("Artifact-rejection threshold diagnostic plot")

    plt.figure(figsize=(10, 6))
    plt.hist(max_peak_to_peak_per_epoch, bins=100, color='skyblue',
             edgecolor='black', alpha=0.7, label='Epoch amplitude')
    plt.axvline(threshold, color='red', linestyle='--', linewidth=2,
                label='Current threshold')
    plt.title('Distribution of Maximum Amplitude per Epoch')
    plt.xlabel('Peak-to-peak amplitude (uV)')
    plt.ylabel('Number of epochs')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


def artifact_detection(data, threshold, num_epochs, samples_per_epoch, show_plot=False):
    """Create a mask excluding artifact-contaminated and final 30 epochs.

    Args:
        data: Signal data arranged as channels by samples.
        threshold: Multiplier applied to the global signal standard deviation.
        num_epochs: Number of complete epochs in the signal.
        samples_per_epoch: Number of samples in each epoch.
        show_plot: Whether to display the amplitude diagnostic plot.

    Returns:
        Boolean mask marking epochs to keep, excluding the final 30.
    """
    bad_epoch_indices = []
    max_peak_to_peak_per_epoch = []

    threshold = threshold * np.std(data)

    for epoch_index in range(num_epochs):
        start = epoch_index * samples_per_epoch
        end = start + samples_per_epoch
        epoch_data = data[:, start:end]

        peak_to_peak = np.ptp(epoch_data, axis=1)
        max_peak_to_peak_per_epoch.append(np.max(peak_to_peak))

        if np.any(peak_to_peak > threshold):
            bad_epoch_indices.append(epoch_index)

    max_peak_to_peak_per_epoch = np.array(max_peak_to_peak_per_epoch)
    bad_epoch_indices = np.array(bad_epoch_indices)
    keep_mask = np.ones(num_epochs, dtype=bool)
    keep_mask[bad_epoch_indices] = False
    keep_mask = keep_mask[:-30]

    print(f"Artifact-contaminated epochs: {len(bad_epoch_indices)} / {num_epochs}")
    print(f"Amplitude threshold: {threshold}")
    if show_plot:
        artifact_graph(threshold, max_peak_to_peak_per_epoch)
    return keep_mask








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


def process_patient(patient_id, channels_to_drop, epoch_length=30,remove_artifacts=True, artifact_threshold=ARTIFACT_THRESHOLD, show_plot=False, base_path=None):
    """Extract sleep features and labels for a patient after preprocessing.

    Args:
        patient_id: Patient identifier.
        channels_to_drop: Channel names to remove from the EDF recording.
        base_path: Directory containing patient subfolders. Defaults to the
            project's data directory.
        epoch_length: Epoch duration in seconds.
        remove_artifacts: Enables artifact filtering when True.
        artifact_threshold: Multiplier used to detect high-amplitude epochs.
        show_plot: Whether to display the artifact diagnostic plot.

    Returns:
        A tuple containing the feature table and the two filtered label arrays.
    """
    edf_path, labels = get_edf_patient(patient_id, base_path=base_path)

    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    raw.drop_channels([channel for channel in channels_to_drop if channel in raw.ch_names])

    raw.resample(100)
    raw.filter(0.3, 49, verbose=False)
    sampling_frequency = raw.info['sfreq']
    samples_per_epoch = int(epoch_length * sampling_frequency)
    data = raw.get_data()
    num_epochs = data.shape[1] // samples_per_epoch

    sleep_staging = yasa.SleepStaging(raw, eeg_name='C4-A1', eog_name='LOC-A2', emg_name='X1')
    features = sleep_staging.get_features()[:-30]

    if remove_artifacts:
        keep_mask = artifact_detection(data, artifact_threshold, num_epochs, samples_per_epoch, show_plot)

        annotator_1_labels = labels[0][keep_mask]
        annotator_2_labels = labels[1][keep_mask]
        features = features[keep_mask]
    else:
        annotator_1_labels = labels[0] 
        annotator_2_labels = labels[1] 
        features = features 

    return features, annotator_1_labels, annotator_2_labels
