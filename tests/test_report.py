import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from report import generate_report


def _probabilities(labels, class_count):
    probabilities = np.full((len(labels), class_count), 0.01)
    probabilities[np.arange(len(labels)), labels] = 0.96
    probabilities += np.random.default_rng(7).random(probabilities.shape) * 0.01
    return probabilities / probabilities.sum(axis=1, keepdims=True)


class ReportGenerationTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(11)
        self.y_multiclass = np.tile(np.arange(4), 10)
        self.probabilities_multiclass = _probabilities(
            self.y_multiclass, class_count=4
        )
        self.y_binary = np.tile(np.arange(2), 15)
        self.probabilities_binary = np.where(self.y_binary == 1, 0.9, 0.1)
        self.y_combined = np.tile(np.arange(5), 12)
        self.probabilities_combined = _probabilities(
            self.y_combined, class_count=5
        )
        self.confidence_mask = rng.random(len(self.y_combined)) > 0.25

        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.output_path = Path(self.temporary_directory.name) / "report.html"
        self.result = generate_report(
            y_true_multiclass=self.y_multiclass,
            y_pred_multiclass=self.y_multiclass,
            probabilities_multiclass=self.probabilities_multiclass,
            y_true_binary=self.y_binary,
            y_pred_binary=self.y_binary,
            probabilities_binary=self.probabilities_binary,
            y_true_combined=self.y_combined,
            y_pred_combined=self.y_combined,
            probabilities_combined=self.probabilities_combined,
            confidence_mask=self.confidence_mask,
            confidence_threshold=0.95,
            artifact_threshold=60,
            train_patient_ids=[1, 2, 3, 4, 5, 6, 7],
            test_patient_ids=[8, 9, 10],
            output_path=self.output_path,
        )
        self.html = self.output_path.read_text(encoding="utf-8")

    def test_writes_the_report_to_the_requested_path(self):
        self.assertEqual(self.result, self.output_path)
        self.assertTrue(self.output_path.exists())

    def test_includes_the_model_sections_and_metric_tables(self):
        for heading in (
            "Modelo multiclase",
            "Modelo binario",
            "Modelo combinado - Sin filtrar",
        ):
            self.assertIn(heading, self.html)

        self.assertIn("Macro-F1", self.html)
        self.assertIn("Resumen de métricas", self.html)
        self.assertIn("Precisión", self.html)

    def test_embeds_roc_and_confusion_matrix_images(self):
        self.assertGreaterEqual(
            self.html.count("data:image/png;base64,"), 6
        )
        self.assertIn("Curva ROC", self.html)
        self.assertIn("Matriz de confusión", self.html)

    def test_includes_split_details_and_thresholds(self):
        self.assertIn(
            "Pacientes de entrenamiento: 1, 2, 3, 4, 5, 6, 7", self.html
        )
        self.assertIn("Pacientes de prueba: 8, 9, 10", self.html)
        self.assertIn("Umbral de detección de artefactos: 60", self.html)
        self.assertIn("Umbral de confianza del combinado: 0.95", self.html)


if __name__ == "__main__":
    unittest.main()
