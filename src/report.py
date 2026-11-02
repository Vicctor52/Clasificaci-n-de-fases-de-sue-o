import base64
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, auc, f1_score, precision_recall_fscore_support, roc_curve

from plots import plot_confusion_matrix, plot_roc


def _calculate_auc(y_true, probabilities, labels):
    """Calculate macro one-vs-rest AUC, or the positive-class AUC for binary data."""
    y_true = np.asarray(y_true)
    probabilities = np.asarray(probabilities)
    labels = np.asarray(labels)

    class_indices = [1] if len(labels) == 2 else range(len(labels))
    class_aucs = []
    for class_index in class_indices:
        binary_labels = (y_true == labels[class_index]).astype(int)
        if np.unique(binary_labels).size < 2:
            continue
        scores = (
            probabilities
            if probabilities.ndim == 1
            else probabilities[:, class_index]
        )
        false_positive_rate, true_positive_rate, _ = roc_curve(
            binary_labels, scores
        )
        class_aucs.append(auc(false_positive_rate, true_positive_rate))

    return float(np.mean(class_aucs)) if class_aucs else None


def _metrics_html(y_true, y_pred, probabilities, labels, class_names):
    """Build summary and per-class metric tables."""
    if not len(y_true):
        return "<p>No samples available to calculate the metrics.</p>"

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    macro_f1 = f1_score(
        y_true, y_pred, labels=labels, average="macro", zero_division=0
    )
    auc_score = _calculate_auc(y_true, probabilities, labels)
    auc_text = f"{auc_score:.3f}" if auc_score is not None else "Unavailable"

    class_rows = "".join(
        f"<tr><td>{name}</td><td>{p:.3f}</td><td>{r:.3f}</td>"
        f"<td>{score:.3f}</td><td>{int(count)}</td></tr>"
        for name, p, r, score, count in zip(
            class_names, precision, recall, f1, support
        )
    )
    return f"""
    <table>
      <thead><tr><th>Samples</th><th>Accuracy</th><th>Macro-F1</th><th>AUC</th></tr></thead>
      <tbody><tr><td>{len(y_true)}</td><td>{accuracy_score(y_true, y_pred):.3f}</td>
      <td>{macro_f1:.3f}</td><td>{auc_text}</td></tr></tbody>
    </table>
    <table>
      <thead><tr><th>Class</th><th>Precision</th><th>Recall</th><th>F1</th><th>Support</th></tr></thead>
      <tbody>{class_rows}</tbody>
    </table>
    """


def _plot_image(plot_function, *args, **kwargs):
    """Render a plot to an HTML-ready PNG image."""
    with patch("matplotlib.pyplot.show"):
        plot_function(*args, **kwargs)
    figure = plt.gcf()
    image = BytesIO()
    figure.savefig(image, format="png", bbox_inches="tight")
    plt.close(figure)
    return base64.b64encode(image.getvalue()).decode("ascii")


def generate_report(
    y_true_multiclass,
    y_pred_multiclass,
    probabilities_multiclass,
    y_true_binary,
    y_pred_binary,
    probabilities_binary,
    y_true_combined,
    y_pred_combined,
    probabilities_combined,
    confidence_mask,
    confidence_threshold,
    artifact_threshold,
    train_patient_ids,
    test_patient_ids,
    output_path=None,
):
    """Generate an HTML report with ROC curves and confusion matrices.

    Args:
        y_true_multiclass: True labels for the four-class model.
        y_pred_multiclass: Predicted labels for the four-class model.
        probabilities_multiclass: Class probabilities for the four-class model.
        y_true_binary: True labels for the binary model.
        y_pred_binary: Thresholded predictions for the binary model.
        probabilities_binary: Positive-class scores for the binary model.
        y_true_combined: True labels for the combined model before filtering.
        y_pred_combined: Predictions for the combined model before filtering.
        probabilities_combined: Class probabilities before confidence filtering.
        confidence_mask: Boolean mask marking accepted combined predictions.
        confidence_threshold: Confidence threshold used by the combined model.
        artifact_threshold: Multiplier used to flag high-amplitude epochs.
        output_path: Optional destination for the HTML report. Defaults to
            ``model_report.html`` in the project root.
        train_patient_ids: Patient IDs used to train the models.
        test_patient_ids: Patient IDs used to evaluate the models.

    Returns:
        The path to the generated HTML report.
    """
    multiclass_names = ["Wake", "N1, N2", "N3", "REM"]
    multiclass_labels = [0, 1, 2, 3]
    binary_names = ["N1", "N2"]
    binary_labels = [0, 1]
    combined_names = ["Wake", "N1", "N2", "N3", "REM"]
    combined_labels = [0, 1, 2, 3, 4]

    confidence_mask = np.asarray(confidence_mask, dtype=bool)
    accepted_count = int(confidence_mask.sum())
    total_count = len(confidence_mask)
    y_true_combined = np.asarray(y_true_combined)
    y_pred_combined = np.asarray(y_pred_combined)
    probabilities_combined = np.asarray(probabilities_combined)

    sections = [
        (
            "Multiclass model",
            "Classifies each epoch as Wake, N1/N2, N3, or REM. The ROC curve "
            "shows the performance of each class against the rest; the "
            "confusion matrix compares the true labels with the predictions.",
            y_true_multiclass,
            y_pred_multiclass,
            probabilities_multiclass,
            multiclass_labels,
            multiclass_names,
            None,
        ),
        (
            "Binary model",
            "Distinguishes between N1 and N2. The ROC curve evaluates the "
            "positive-class scores, and the confusion matrix shows predictions "
            "at the selected threshold.",
            y_true_binary,
            y_pred_binary,
            probabilities_binary,
            binary_labels,
            binary_names,
            None,
        ),
        (
            "Combined model - Unfiltered",
            "Combines the multiclass and binary models to classify the five "
            "sleep stages. The plots include all samples.",
            y_true_combined,
            y_pred_combined,
            probabilities_combined,
            combined_labels,
            combined_names,
            None,
        ),
        (
            "Combined model - Confidence-filtered",
            f"Evaluates only predictions that exceed the confidence threshold "
            f"of {confidence_threshold:.2f}: {accepted_count}/{total_count} "
            f"samples ({accepted_count / total_count:.1%} coverage). Both the "
            "ROC curve and confusion matrix use this same subset.",
            y_true_combined,
            y_pred_combined,
            probabilities_combined,
            combined_labels,
            combined_names,
            confidence_mask,
        ),
    ]

    report_sections = []
    combined_section_open = False
    for (
        name,
        description,
        y_true,
        y_pred,
        probabilities,
        labels,
        class_names,
        sample_mask,
    ) in sections:
        selected_true = np.asarray(y_true)
        selected_pred = np.asarray(y_pred)
        selected_probabilities = np.asarray(probabilities)
        if sample_mask is not None:
            selected_true = selected_true[sample_mask]
            selected_pred = selected_pred[sample_mask]
            selected_probabilities = selected_probabilities[sample_mask]

        metrics_content = _metrics_html(
            selected_true,
            selected_pred,
            selected_probabilities,
            labels,
            class_names,
        )

        if selected_true.size:
            has_roc_data = any(
                np.unique(selected_true == label).size == 2 for label in labels
            )
            if has_roc_data:
                roc_image = _plot_image(
                    plot_roc,
                    selected_true,
                    selected_probabilities,
                    labels=labels,
                    class_names=class_names,
                    title=f"ROC curve - {name}",
                )
                roc_content = (
                    f'<img src="data:image/png;base64,{roc_image}" '
                    f'alt="ROC curve - {name}">'
                )
            else:
                roc_content = (
                    "<p>No classes have enough positive and negative samples "
                    "to calculate the ROC curve.</p>"
                )

            matrix_image = _plot_image(
                plot_confusion_matrix,
                selected_true,
                selected_pred,
                labels=labels,
                class_names=class_names,
                title=f"Confusion matrix - {name}",
            )
            matrix_content = (
                f'<img src="data:image/png;base64,{matrix_image}" '
                f'alt="Confusion matrix - {name}">'
            )
        else:
            roc_content = "<p>No samples available to calculate the ROC curve.</p>"
            matrix_content = (
                "<p>No samples available to calculate the confusion matrix.</p>"
            )

        is_combined = name.startswith("Combined model - ")
        if is_combined:
            if not combined_section_open:
                report_sections.append('<section><h2>Combined model</h2>')
                combined_section_open = True
            subsection_name = name.removeprefix("Combined model - ")
            section_open = '<div class="subsection">'
            section_heading = f"<h3>{subsection_name}</h3>"
            section_close = "</div>"
        else:
            section_open = "<section>"
            section_heading = f"<h2>{name}</h2>"
            section_close = "</section>"

        report_sections.append(
            f"""
            {section_open}
              {section_heading}
              <p>{description}</p>
              <h4>Metric summary</h4>
              {metrics_content}
              <div class="plots">
                <figure>
                {roc_content}
                  <figcaption>ROC curve</figcaption>
                </figure>
                <figure>
                  {matrix_content}
                  <figcaption>Confusion matrix</figcaption>
                </figure>
              </div>
            {section_close}
            """
        )

    if combined_section_open:
        report_sections.append("</section>")

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Model Evaluation Report</title>
  <style>
    body {{ font-family: Arial, sans-serif; line-height: 1.5; margin: 2rem auto; max-width: 1200px; padding: 0 1rem; color: #222; }}
    h1, h2 {{ color: #17365d; }}
    section {{ border-top: 1px solid #ddd; margin-top: 2rem; padding-top: 1rem; }}
    .subsection {{ border-top: 1px solid #eee; margin-top: 1.5rem; padding-top: 1rem; }}
    h3 {{ color: #345; }}
    .plots {{ display: flex; flex-wrap: wrap; gap: 1rem; }}
    figure {{ flex: 1 1 420px; margin: 0; }}
    img {{ height: auto; max-width: 100%; }}
    figcaption {{ font-weight: bold; text-align: center; }}
    table {{ border-collapse: collapse; margin: 0.75rem 0 1.25rem; width: auto; }}
    th, td {{ border: 1px solid #ccc; padding: 0.4rem 0.7rem; text-align: right; }}
    th:first-child, td:first-child {{ text-align: left; }}
    th {{ background: #f0f3f7; }}
  </style>
</head>
<body>
  <h1>Model Evaluation Report</h1>
  <p>This report summarizes the performance of the sleep-stage classification
  models using ROC curves and confusion matrices. The ROC curve evaluates
  discrimination across thresholds; the matrix shows correct and incorrect
  final predictions.</p>
  <section>
    <h2>Experiment Summary</h2>
    <ul>
      <li>Training patients: {", ".join(map(str, train_patient_ids))}.</li>
      <li>Test patients: {", ".join(map(str, test_patient_ids))}.</li>
      <li>Only epochs with matching labels from both annotators are retained.</li>
      <li>Artifact detection threshold: {artifact_threshold} × signal standard deviation.</li>
      <li>Epochs evaluated: {len(y_true_multiclass)} for the multiclass model,
      {len(y_true_binary)} for the binary model, and {total_count} for the combined model.</li>
      <li>Combined-model confidence threshold: {confidence_threshold:.2f};
      {accepted_count} of {total_count} epochs were accepted
      ({accepted_count / total_count:.1%} coverage).</li>
    </ul>
  </section>
  {''.join(report_sections)}
</body>
</html>
"""

    destination = (
        Path(output_path)
        if output_path
        else Path(__file__).resolve().parent.parent / "model_report.html"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(html, encoding="utf-8")
    print(f"Report saved to: {destination.resolve()}")
    return destination
