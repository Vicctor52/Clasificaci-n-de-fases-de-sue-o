import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import utils


class GetEdfPatientTests(unittest.TestCase):
    def test_loads_both_annotators_and_maps_rem_to_four(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            patient_folder = Path(temporary_directory) / "1"
            patient_folder.mkdir()

            annotator_1 = [0, 1, 2, 3, 5] + [0] * 30
            annotator_2 = [0, 2, 2, 4, 5] + [1] * 30
            (patient_folder / "1_1.txt").write_text(
                "\n".join(map(str, annotator_1)) + "\n"
            )
            (patient_folder / "1_2.txt").write_text(
                "\n".join(map(str, annotator_2)) + "\n"
            )

            edf_path, labels = utils.get_edf_patient(
                1, base_path=temporary_directory
            )

            self.assertEqual(edf_path, patient_folder / "1.edf")
            self.assertEqual(len(labels), 2)
            self.assertEqual(labels[0].tolist(), [0, 1, 2, 3, 4])
            self.assertEqual(labels[1].tolist(), [0, 2, 2, 4, 4])


class ArtifactDetectionTests(unittest.TestCase):
    def test_marks_high_amplitude_epochs_and_drops_the_last_thirty(self):
        samples_per_epoch = 10
        num_epochs = 40
        data = np.zeros((1, num_epochs * samples_per_epoch))
        data[0, :samples_per_epoch] = np.arange(samples_per_epoch)

        keep_mask = utils.artifact_detection(
            data,
            threshold=0.5,
            num_epochs=num_epochs,
            samples_per_epoch=samples_per_epoch,
        )

        self.assertEqual(len(keep_mask), num_epochs - 30)
        self.assertFalse(keep_mask[0])
        self.assertEqual(int(keep_mask.sum()), num_epochs - 31)

    def test_ignores_artifacts_within_the_final_thirty_epochs(self):
        samples_per_epoch = 10
        num_epochs = 40
        data = np.zeros((1, num_epochs * samples_per_epoch))
        data[0, 35 * samples_per_epoch:36 * samples_per_epoch] = np.arange(
            samples_per_epoch
        )

        keep_mask = utils.artifact_detection(
            data,
            threshold=0.5,
            num_epochs=num_epochs,
            samples_per_epoch=samples_per_epoch,
        )

        self.assertTrue(np.all(keep_mask))


if __name__ == "__main__":
    unittest.main()
