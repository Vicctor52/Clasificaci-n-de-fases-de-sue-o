import sys
import unittest
from pathlib import Path

import mne
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import utils
from config import ARTIFACT_THRESHOLD, CHANNELS_TO_DROP, CONFIDENCE_THRESHOLD
from model_utils import get_confidence_mask

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

CONFIDENCE_RESULTS = {
    0.8: {
        "coverage": 0.7406849830636055,
        "macro_f1": 0.9004519162690123,
        "accuracy": 0.9115853658536586,
        "auc": 0.9800052831546846,
    },
    0.85: {
        "coverage": 0.6872412495295446,
        "macro_f1": 0.9152428872982282,
        "accuracy": 0.9282584884994524,
        "auc": 0.980767692155314,
    },
    0.9: {
        "coverage": 0.6164847572450132,
        "macro_f1": 0.9240990244198933,
        "accuracy": 0.9438339438339438,
        "auc": 0.9812349579165447,
    },
    0.95: {
        "coverage": 0.502822732404968,
        "macro_f1": 0.9351303966512964,
        "accuracy": 0.9610778443113772,
        "auc": 0.9822363951643471,
    },
}

UNFILTERED_RESULTS = {
    "coverage": 1.0,
    "macro_f1": 0.8255179774164318,
    "accuracy": 0.8231087692886714,
    "auc": 0.9658341664634387,
}


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


class ConfidenceThresholdTests(unittest.TestCase):
    def test_configured_threshold_is_one_of_the_evaluated_candidates(self):
        self.assertIn(CONFIDENCE_THRESHOLD, CONFIDENCE_RESULTS)

    def test_selected_threshold_maximizes_macro_f1_with_minimum_coverage(self):
        eligible = {
            threshold: result
            for threshold, result in CONFIDENCE_RESULTS.items()
            if result["coverage"] >= 0.5
        }
        selected = max(
            eligible, key=lambda threshold: eligible[threshold]["macro_f1"]
        )

        self.assertEqual(CONFIDENCE_THRESHOLD, selected)
        self.assertGreaterEqual(
            CONFIDENCE_RESULTS[CONFIDENCE_THRESHOLD]["coverage"], 0.5
        )

    def test_filtered_macro_f1_improves_over_the_unfiltered_model(self):
        self.assertGreater(
            CONFIDENCE_RESULTS[CONFIDENCE_THRESHOLD]["macro_f1"],
            UNFILTERED_RESULTS["macro_f1"],
        )

    def test_higher_thresholds_reduce_coverage(self):
        coverages = [
            CONFIDENCE_RESULTS[threshold]["coverage"]
            for threshold in sorted(CONFIDENCE_RESULTS)
        ]
        self.assertEqual(coverages, sorted(coverages, reverse=True))

    def test_mask_is_strict_at_the_threshold_boundary(self):
        probabilities = np.array(
            [
                [0.95, 0.05],
                [0.9500001, 0.0499999],
                [0.9499999, 0.0500001],
            ]
        )

        mask = get_confidence_mask(probabilities, threshold=0.95)

        np.testing.assert_array_equal(mask, [False, True, False])


if __name__ == "__main__":
    unittest.main()
