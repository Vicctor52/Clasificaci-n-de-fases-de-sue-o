import sys
import unittest
from pathlib import Path

import mne
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import utils
from config import ARTIFACT_THRESHOLD, CHANNELS_TO_DROP

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


class ArtifactThresholdTests(unittest.TestCase):
    def test_configured_threshold_is_the_validated_value(self):
        self.assertEqual(ARTIFACT_THRESHOLD, 60)

    def test_rejects_a_large_artifact_and_keeps_a_physiological_slow_wave(self):
        samples_per_epoch = 100
        num_epochs = 1000
        generator = np.random.default_rng(42)
        data = generator.normal(
            loc=0.0,
            scale=0.1,
            size=(1, num_epochs * samples_per_epoch),
        )

        slow_wave_index = 100
        slow_wave = 2 * np.sin(
            np.linspace(0, 2 * np.pi, samples_per_epoch)
        )
        data[0, slow_wave_index * samples_per_epoch:(slow_wave_index + 1) * samples_per_epoch] = slow_wave

        artifact_index = 200
        artifact = np.tile([20.0, -20.0], samples_per_epoch // 2)
        data[0, artifact_index * samples_per_epoch:(artifact_index + 1) * samples_per_epoch] = artifact

        keep_mask = utils.artifact_detection(
            data,
            threshold=ARTIFACT_THRESHOLD,
            num_epochs=num_epochs,
            samples_per_epoch=samples_per_epoch,
        )

        self.assertEqual(len(keep_mask), num_epochs - 30)
        self.assertTrue(keep_mask[slow_wave_index])
        self.assertFalse(keep_mask[artifact_index])

    @unittest.skipUnless(DATA_DIR.is_dir(), "ISRUC-Sleep data is not available")
    def test_rejection_rate_on_the_dataset_is_within_the_operating_range(self):
        rejected_epochs = 0
        evaluated_epochs = 0

        for patient_id in range(1, 11):
            edf_path = DATA_DIR / str(patient_id) / f"{patient_id}.edf"
            raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
            raw.drop_channels(
                [channel for channel in CHANNELS_TO_DROP if channel in raw.ch_names]
            )
            raw.resample(100)
            raw.filter(0.3, 49, verbose=False)

            sampling_frequency = raw.info["sfreq"]
            samples_per_epoch = int(30 * sampling_frequency)
            data = raw.get_data()
            num_epochs = data.shape[1] // samples_per_epoch

            keep_mask = utils.artifact_detection(
                data,
                threshold=ARTIFACT_THRESHOLD,
                num_epochs=num_epochs,
                samples_per_epoch=samples_per_epoch,
            )
            rejected_epochs += int((~keep_mask).sum())
            evaluated_epochs += len(keep_mask)

        rejection_rate = rejected_epochs / evaluated_epochs
        self.assertGreaterEqual(rejection_rate, 0.005)
        self.assertLessEqual(rejection_rate, 0.02)


if __name__ == "__main__":
    unittest.main()
