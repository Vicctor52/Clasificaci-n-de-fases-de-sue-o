import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from model_utils import (
    binary_data_preparation,
    binary_xgb_evaluation,
    data_preparation,
    get_confidence_mask,
    model_combination,
)


class _MockMulticlassModel:
    def __init__(self, probabilities):
        self.probabilities = np.asarray(probabilities)

    def predict_proba(self, X):
        return self.probabilities


class _MockBinaryModel:
    def __init__(self, probabilities_by_row):
        self.probabilities_by_row = probabilities_by_row

    def predict_proba(self, X):
        row_position = X.index[0]
        return np.asarray([self.probabilities_by_row[row_position]])


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


class BinaryPreparationTests(unittest.TestCase):
    def test_keeps_only_n1_and_n2_and_remaps_their_labels(self):
        X_train = pd.DataFrame({"feature": [1, 2, 3, 4]})
        X_test = pd.DataFrame({"feature": [5, 6, 7]})

        X_train_bin, y_train_bin, X_test_bin, y_test_bin = (
            binary_data_preparation(
                X_train,
                X_test,
                np.array([0, 1, 2, 3]),
                np.array([4, 1, 2]),
            )
        )

        self.assertEqual(X_train_bin["feature"].tolist(), [2, 3])
        self.assertEqual(X_test_bin["feature"].tolist(), [6, 7])
        np.testing.assert_array_equal(y_train_bin, [0, 1])
        np.testing.assert_array_equal(y_test_bin, [0, 1])


class _FakeBinaryModel:
    def __init__(self, probabilities):
        self.probabilities = probabilities

    def predict_proba(self, X_test):
        return self.probabilities


class BinaryEvaluationTests(unittest.TestCase):
    def test_thresholds_probabilities_and_plots_the_binary_results(self):
        y_test_bin = np.array([0, 1, 0, 1])
        probabilities = np.array(
            [
                [0.90, 0.10],
                [0.20, 0.80],
                [0.40, 0.60],
                [0.10, 0.90],
            ]
        )

        with patch("model_utils.plot_confusion_matrix") as confusion_plot, patch(
            "model_utils.plot_roc", return_value={"N2": 0.9}
        ) as roc_plot:
            binary_xgb_evaluation(
                _FakeBinaryModel(probabilities),
                np.zeros((4, 2)),
                y_test_bin,
                threshold=0.5,
            )

        confusion_args, confusion_kwargs = confusion_plot.call_args
        np.testing.assert_array_equal(confusion_args[0], y_test_bin)
        np.testing.assert_array_equal(confusion_args[1], np.array([0, 1, 1, 1]))
        self.assertEqual(confusion_kwargs["labels"], [0, 1])
        self.assertEqual(confusion_kwargs["class_names"], ["N1", "N2"])

        roc_args, roc_kwargs = roc_plot.call_args
        np.testing.assert_array_equal(roc_args[0], y_test_bin)
        np.testing.assert_allclose(roc_args[1], probabilities[:, 1])
        self.assertEqual(roc_kwargs["labels"], [0, 1])


class ModelCombinationTests(unittest.TestCase):
    def setUp(self):
        self.X_test = pd.DataFrame({"feature": [1.0, 2.0]})
        self.multiclass_model = _MockMulticlassModel(
            [
                [0.9, 0.02, 0.04, 0.04],
                [0.05, 0.6, 0.1, 0.25],
            ]
        )
        self.binary_model = _MockBinaryModel(
            {
                0: [0.9, 0.1],
                1: [0.2, 0.8],
            }
        )

    def test_combines_the_binary_light_sleep_split_with_multiclass_probabilities(
        self,
    ):
        predictions, probabilities, mask = model_combination(
            self.X_test,
            self.multiclass_model,
            self.binary_model,
        )

        np.testing.assert_array_equal(predictions, [0, 2])
        np.testing.assert_allclose(
            probabilities,
            [
                [0.9, 0.018, 0.002, 0.04, 0.04],
                [0.05, 0.12, 0.48, 0.1, 0.25],
            ],
        )
        np.testing.assert_array_equal(mask, [True, True])

    def test_filters_rows_below_the_confidence_threshold(self):
        predictions, probabilities, mask = model_combination(
            self.X_test,
            self.multiclass_model,
            self.binary_model,
            confidence_threshold=0.8,
        )

        np.testing.assert_array_equal(mask, [True, False])
        np.testing.assert_array_equal(predictions, [0])
        np.testing.assert_allclose(
            probabilities,
            [[0.9, 0.018, 0.002, 0.04, 0.04]],
        )

    def test_confidence_mask_uses_the_maximum_class_probability(self):
        probabilities = np.array(
            [
                [0.75, 0.25],
                [0.2, 0.8],
                [0.5, 0.5],
            ]
        )

        mask = get_confidence_mask(probabilities, threshold=0.75)

        np.testing.assert_array_equal(mask, [False, True, False])


if __name__ == "__main__":
    unittest.main()
