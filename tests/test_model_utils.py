import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from model_utils import data_preparation


class DataPreparationTests(unittest.TestCase):
    def test_drops_disagreements_and_remaps_light_sleep_labels(self):
        data_list = [
            {
                "id": 1,
                "X": pd.DataFrame({"feature": [1, 2, 3, 4]}),
                "y1": np.array([0, 1, 2, 3]),
                "y2": np.array([0, 1, 1, 3]),
            },
            {
                "id": 2,
                "X": pd.DataFrame({"feature": [5, 6, 7]}),
                "y1": np.array([4, 2, 1]),
                "y2": np.array([4, 2, 0]),
            },
        ]

        (
            X_train,
            X_test,
            y_train,
            y_test,
            y_train_original,
            y_test_original,
        ) = data_preparation(data_list, train_patients=1)

        self.assertEqual(X_train["feature"].tolist(), [1, 2, 4])
        self.assertEqual(X_test["feature"].tolist(), [5, 6])
        np.testing.assert_array_equal(y_train_original, [0, 1, 3])
        np.testing.assert_array_equal(y_test_original, [4, 2])
        np.testing.assert_array_equal(y_train, [0, 1, 2])
        np.testing.assert_array_equal(y_test, [3, 1])

    def test_concatenates_patients_in_the_same_split_and_resets_the_index(self):
        data_list = [
            {
                "id": 1,
                "X": pd.DataFrame({"feature": [1, 2]}),
                "y1": np.array([0, 1]),
                "y2": np.array([0, 1]),
            },
            {
                "id": 2,
                "X": pd.DataFrame({"feature": [3]}),
                "y1": np.array([2]),
                "y2": np.array([2]),
            },
            {
                "id": 3,
                "X": pd.DataFrame({"feature": [4]}),
                "y1": np.array([3]),
                "y2": np.array([3]),
            },
        ]

        (
            X_train,
            X_test,
            y_train,
            y_test,
            _,
            _,
        ) = data_preparation(data_list, train_patients=2)

        self.assertEqual(X_train["feature"].tolist(), [1, 2, 3])
        self.assertEqual(X_train.index.tolist(), [0, 1, 2])
        self.assertEqual(X_test["feature"].tolist(), [4])
        np.testing.assert_array_equal(y_train, [0, 1, 1])
        np.testing.assert_array_equal(y_test, [2])


if __name__ == "__main__":
    unittest.main()
