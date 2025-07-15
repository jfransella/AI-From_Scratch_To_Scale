# **Codebase Architecture & Strategy**

## **1\. Guiding Philosophy: A Scalable & Focused Architecture**

This architecture is designed to support the project's core mission: "learning by building." It achieves this by separating **shared, reusable infrastructure** from **unique, model-specific code**. The primary goals are to focus on learning, maximize consistency, and minimize repetition.

## **2\. Top-Level Directory Structure**

The repository will be organized with shared packages and configuration at the root level.

/ai-from-scratch-to-scale/  
|  
|-- /.github/             \# Contains GitHub-specific files for community management.  
|-- /data\_utils/          \# SHARED: Handles all dataset loading & transformations.  
|-- /docs/                \# SHARED: Project-wide documentation (e.g., this file).  
|-- /engine/              \# SHARED: Reusable training/evaluation engine with wandb.  
|-- /plotting/            \# SHARED: Generates and saves visualization files.  
|-- /tests/               \# SHARED: Automated tests for the project.  
|-- /utils/               \# SHARED: General-purpose utilities (logging, seeds, etc.).  
|  
|-- /models/              \# Contains all individual model projects.  
|   |-- /01\_perceptron/  
|   |-- ...  
|  
|-- .gitignore            \# Defines ignored files for the entire project.  
|-- LICENSE  
|-- README.md             \# The main project README, links to /docs.  
|-- requirements-dev.txt  \# Development-only dependencies (pytest, black, flake8).

## **3\. The Shared Packages & Configuration**

### **/.github**

* **Purpose:** To improve project management and standardize community interaction.  
* **Contents:**  
  * ISSUE\_TEMPLATE/: Templates for bug reports and feature requests.  
  * PULL\_REQUEST\_TEMPLATE.md: A checklist and guide for new pull requests.  
  * workflows/: GitHub Actions for Continuous Integration (e.g., running black for formatting checks, flake8 for linting, and pytest for testing).  
  * CONTRIBUTING.md: Guidelines for how others can contribute.  
  * dependabot.yml: Configuration for automated dependency updates.  
  * copilot-instructions.md: Context for GitHub Copilot to improve its suggestions within the repo.

### **/data\_utils, /engine, /plotting, /utils**

These packages contain the shared, reusable code for data handling, training, visualization, and general utilities, respectively.

### **3.1. Dependency & Environment Management**

To ensure historical accuracy and prevent dependency conflicts, we will adopt a **per-model virtual environment** strategy.

* **Model Dependencies (requirements.txt):** Each model project inside /models/ will have its own requirements.txt file, listing only the specific libraries and versions needed for that model.  
* **Development Dependencies (requirements-dev.txt):** A single requirements-dev.txt file at the project root will define dependencies needed only for development and testing: pytest for testing, black for code formatting, and flake8 for linting.  
* **Workflow for Shared Code:** To work on or test the shared packages, you will use a model's virtual environment as the "host." The process is:  
  1. Activate a chosen model's virtual environment.  
  2. Install the development dependencies using pip install \-r requirements-dev.txt from the project root.  
  3. Install the project in editable mode using pip install \-e . from the project root. This makes the shared packages available within the active environment.

### **3.2. Testing Strategy**

To ensure the reliability and correctness of the shared infrastructure, we will adopt a formal testing strategy using the **pytest** framework.

* **Unit Tests:** These will test individual functions in the shared packages (/utils, /plotting, /data\_utils) in isolation.  
* **Integration Tests:** These will test that the core components work together. The most important will be a "smoke test" for the /engine that runs a dummy model for a single epoch to ensure the entire training pipeline is functional.  
* **Automation (CI):** The GitHub Actions workflow (/.github/workflows/ci.yml) will be configured to run all tests automatically on every pull request, acting as a quality gate to prevent regressions.

## **4\. Dual-System Logging Strategy**

To capture both a human-readable narrative and structured metrics, we will use two systems in parallel, orchestrated by the /engine.

* **System 1: Python logging (The Narrative Log)**  
* **System 2: Weights & Biases (wandb) (The Metrics Database)**

## **5\. Visualization Strategy**

This strategy separates visualization **generation** from **display** to support automated runs, wandb integration, and interactive notebook analysis.

* **Activation**: Generation is triggered by the \--visualize command-line flag.  
* **Selection Logic**: The model-specific **train.py script is responsible for deciding *which* visualizations to generate**. It contains a static mapping that links an \--experiment name to a list of required plot function names.  
* **Execution**: The /engine receives the list of plot names and calls the corresponding functions from /plotting to save the image files to the local /outputs/visualizations directory.

## **6\. Model Checkpointing & Loading**

Saving a model checkpoint after a training run is the default, mandatory behavior to ensure results are preserved. Loading a checkpoint is an optional input for fine-tuning or evaluation.

* **Primary Method: wandb Artifacts**: When wandb is enabled, the trained model's state\_dict will be saved as a versioned wandb Artifact.  
* **Local Fallback**: If wandb is disabled, the model's state\_dict will be saved to the local /outputs/models/ directory.

## **7\. Dedicated Evaluation Workflow**

To cleanly separate the act of training from analysis, each model will have a dedicated evaluate.py script.

* **Purpose**: To load a pre-trained model checkpoint and evaluate its performance on a specified test dataset.  
* **Core Logic**: The script will instantiate the model, load the checkpoint, fetch the test data, and use the Evaluator class from the shared /engine to compute and log final metrics.  
* **Key Arguments**: The script will be driven by arguments like \--checkpoint (required, path or wandb ID), \--experiment (required, to select the dataset), and \--visualize (optional).

## **8\. The Model-Specific Project Structure**

Each model's directory is a self-contained project with its own environment and dependencies.

/models/01\_perceptron/  
|-- /docs/  
|-- /notebooks/  
|-- /outputs/  
|-- /src/  
|   |-- \_\_init\_\_.py  
|   |-- constants.py  
|   |-- config.py  
|   |-- model.py  
|   |-- train.py  
|   |-- evaluate.py       \# New evaluation script  
|-- .venv/                \# Model-specific virtual environment (in .gitignore).  
|-- requirements.txt      \# Model-specific dependencies.  
|-- README.md

## **9\. Parameterized Experiment Execution**

The train.py script is the "Execution Interface" for a run, configured via command-line arguments.

| Argument | Description |
| :---- | :---- |
| \--experiment | **(Required)** The name of the experiment (e.g., xor, iris-hard). |
| \--epochs | Overrides the default number of epochs. |
| \--load-checkpoint | Path or wandb artifact ID to load model weights from before training. |
| \--no-save-checkpoint | A flag to prevent saving the final model checkpoint, useful for debugging. |
| \--no-wandb | A flag to disable wandb logging for the run. |
| \--seed | Sets the random seed for reproducibility. |
| \--tags | Attaches tags to the wandb run for easy filtering. |
| \--visualize | A flag to enable the generation and saving of standard visualizations. |

## **10\. The End-to-End Workflow in Practice**

1. **Setup**: Navigate to a model's directory, create its virtual environment, activate it, and run pip install \-r requirements.txt. For development on shared packages, also run pip install \-r ../../requirements-dev.txt and pip install \-e ../../ from the model directory.  
2. **Training**: Run the train.py command with desired arguments. The script calls helpers from /utils, fetches data, instantiates the model, and passes everything to the Trainer in /engine. The Trainer runs the training loop, logging metrics and saving the final model checkpoint by default.  
3. **Evaluation**: After training, run the evaluate.py command, pointing to the saved checkpoint with \--checkpoint. This script will load the model and use the Evaluator from the /engine to compute and log final performance metrics on the test set.  
4. **Analysis**: The generated logs, visualizations, and metrics from both training and evaluation runs are now available on disk and in wandb, ready for analysis in the project's notebooks.

