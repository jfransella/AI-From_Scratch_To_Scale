# Visualization Implementation Guide

This guide provides step-by-step instructions for integrating visualizations into model training and evaluation
workflows. It ensures consistency, reproducibility, and ease of use across all models in the project.

---

## 1. Integrating Visualization Hooks

- **Add a `--visualize` flag** to all training and evaluation scripts (e.g., `train.py`, `evaluate.py`).
- **Parse the flag** using your argument parser (e.g., argparse).
- **Determine required plots** for the current experiment using the mapping in `config.py` or the Playbooks.
- **Call the appropriate functions** from the `/plotting` package at the end of training/evaluation, passing all
necessary data (e.g., losses, predictions, model, etc.).

---

## 2. Using the /plotting Package

- **Import plotting utilities** at the top of your script:

  ```python
  from plotting import plot_learning_curve, plot_confusion_matrix, ...
  ```text`n- **Follow naming conventions** and docstring standards as described in `Coding_Standards.md`.

- **Pass all required arguments** (e.g., model, data, save_path) to each plotting function.
- **Set random seeds** for reproducibility in plots involving randomness (e.g., t-SNE, UMAP).

---

## 3. Saving and Organizing Output Figures

- **Save all plots** to the model’s `outputs/visualizations/` directory.
- **Use descriptive filenames** (e.g., `learning_curve.png`, `confusion_matrix.png`, `decision_boundary.png`).
- **Do not display plots interactively** in scripts by default; only display in notebooks.
- **Log plots to wandb** if enabled, using the appropriate wandb API calls.

---

## 4. Notebook Integration

- **Include code cells** in analysis notebooks to generate and display all required visualizations.
- **Add markdown cells** interpreting each plot and linking to the Playbooks for reference.

---

## 5. Extending the Visualization Suite

- **Add new visualization functions** to the `/plotting` package with clear, reusable APIs and full docstrings.
- **Update the Playbooks and this guide** whenever a new plot type is added.

---

For a summary of required visualizations by model type, see `docs/visualization/Playbooks.md`. For interpretive
guidance, see `docs/strategy/Visualization_Ideas.md`.
