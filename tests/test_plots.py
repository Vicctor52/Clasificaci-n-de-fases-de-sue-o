import sys
import unittest
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from plots import plot_confusion_matrix, plot_roc


class PlotRocTests(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def test_binary_roc_returns_the_positive_class_auc(self):
        class_aucs = plot_roc(
            np.array([0, 0, 1, 1]),
            np.array([0.1, 0.2, 0.8, 0.9]),
            labels=[0, 1],
            class_names=["N1", "N2"],
            show=False,
        )

        self.assertIn("N2", class_aucs)
        self.assertAlmostEqual(class_aucs["N2"], 1.0)
        axis = plt.gca()
        self.assertEqual(axis.get_xlabel(), "False positive rate")
        self.assertEqual(axis.get_ylabel(), "True positive rate")
        self.assertEqual(axis.get_title(), "ROC curve")
        self.assertIn(
            "ROC curve for N2 (AUC = 1.0000)",
            [text.get_text() for text in axis.get_legend().get_texts()],
        )

    def test_multiclass_roc_returns_one_auc_per_class_and_the_macro_average(self):
        y_true = np.array([0, 1, 2, 3, 0, 1, 2, 3])
        probabilities = np.array(
            [
                [0.80, 0.10, 0.05, 0.05],
                [0.10, 0.70, 0.10, 0.10],
                [0.05, 0.10, 0.75, 0.10],
                [0.05, 0.10, 0.10, 0.75],
                [0.70, 0.15, 0.10, 0.05],
                [0.15, 0.65, 0.10, 0.10],
                [0.10, 0.15, 0.65, 0.10],
                [0.10, 0.10, 0.15, 0.65],
            ]
        )

        class_aucs = plot_roc(
            y_true,
            probabilities,
            labels=[0, 1, 2, 3],
            class_names=["Wake", "N1, N2", "N3", "REM"],
            show=False,
        )

        self.assertEqual(
            set(class_aucs),
            {"Wake", "N1, N2", "N3", "REM", "macro_auc"},
        )
        self.assertGreater(class_aucs["macro_auc"], 0.9)


class PlotConfusionMatrixTests(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def test_returns_the_confusion_matrix_in_label_order(self):
        matrix = plot_confusion_matrix(
            np.array([0, 1, 1, 2]),
            np.array([0, 1, 2, 2]),
            labels=[0, 1, 2],
            class_names=["Wake", "N1", "N2"],
            show=False,
        )

        np.testing.assert_array_equal(
            matrix,
            np.array([[1, 0, 0], [0, 1, 1], [0, 0, 1]]),
        )
        axis = plt.gca()
        self.assertEqual(axis.get_xlabel(), "Predicted label")
        self.assertEqual(axis.get_ylabel(), "True label")
        self.assertEqual(axis.get_title(), "Confusion matrix")


if __name__ == "__main__":
    unittest.main()
