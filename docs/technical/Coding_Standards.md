# **Coding Standards Guide for 'AI From Scratch to Scale'**

## **1\. Purpose**

This document defines the coding standards, conventions, and best practices to be used throughout the "AI From Scratch
to Scale" project. The goal is to ensure all code is consistent, readable, maintainable, and reusable across all 25
model implementations. This guide is a mandatory companion to the Project Charter and the Learning & Development
Strategy.

## **2\. Guiding Philosophy: Code as a Learning Tool**

Our code is not just an implementation; it is a primary learning artifact for both ourselves and the community. It must
be exceptionally clear and well-documented to serve this purpose. We will prioritize clarity and explicit design over
overly clever or compact code.

## **3\. Project Structure**

Every model project will be a new directory within our main GitHub repository and will follow this standardized
structure:

/ai-from\_scratch\_to\_scale/  
|  
|-- /data\_utils/          \# SHARED: Handles all dataset loading  
|-- /engine/              \# SHARED: Reusable training engine with wandb integration  
|-- /plotting/            \# SHARED: Generates visualizations for wandb  
|  
|-- /models/              \# All individual model projects  
|   |  
|   |-- /01\_perceptron/  
|       |-- /data/  
|       |-- /docs/  
|       |-- /notebooks/  
|       |-- /outputs/       \# Local outputs (e.g., final model file)  
|       |-- /src/  
|           |-- \_\_init\_\_.py  
|           |-- constants.py    \# NEW: Fixed values for this project  
|           |-- config.py       \# Hyperparameters for a specific run  
|           |-- model.py        \# The unique Perceptron architecture  
|           |-- train.py        \# Imports and uses the shared engine  
|           |-- evaluate.py     \# Imports and uses the shared engine  
|       |-- requirements.txt  \# Now includes the 'wandb' package  
|       |-- README.md  
|  
|-- LICENSE

## **4\. Coding Style & Formatting**

* **Standard**: All Python code will adhere to the **PEP 8** style guide.  
* **Auto-formatting**: We will use the black code formatter to ensure consistent formatting. Line length will be set to
88 characters.
* **Linting**: We will use a linter like flake8 to catch common errors and style issues.  
* **Imports**: Imports will be grouped in the standard order: 1\) standard library, 2\) third-party libraries, 3\)
local application-specific libraries. They should be sorted alphabetically within each group.

## **5\. Naming Conventions**

### **Directory and File Naming**

* **Model Directories**: `XX_modelname/` format where XX is two-digit number and modelname is lowercase
  * Examples: `01_perceptron/`, `02_adaline/`, `03_mlp/`
  * **Rationale**: Consistent with Unix/Linux conventions, PEP 8 compliant, easier to type
* **Python Files**: snake_case (e.g., `model.py`, `train.py`, `config.py`)
* **Documentation Files**: Title case with underscores (`README.md`, `Implementation_Guide.md`)
* **Notebooks**: Numbered with descriptive names (`01_Theory_and_Intuition.ipynb`)

### **Code Naming**

* **Variables & Functions**: snake_case (e.g., learning_rate, calculate_loss).  
* **Classes**: PascalCase (e.g., Perceptron, ConvolutionalLayer).  
* **Constants**: ALL_CAPS (e.g., LEARNING_RATE, BATCH_SIZE in config.py).  
* **Clarity over Brevity**: Variable names should be descriptive (e.g., use learning_rate instead of lr).

### **Virtual Environment Strategy**

* **Early models** (01_perceptron, 02_adaline, 03_mlp): Use `01_perceptron/.venv` for shared development
* **Later models**: Individual virtual environments as complexity increases
* **Rationale**: Reduces setup overhead for foundational models with similar dependencies

## **6\. Documentation & Commenting**

* **README.md**: Every model project must have a README.md file that serves as the central hub. It will provide a
high-level summary, key takeaways, setup instructions, and links to the detailed documentation in the /docs directory.
* **Docstrings**: All modules, functions, and classes must have a docstring explaining their purpose, arguments, and
what they return. We will use the **Google Python Style Guide** format for docstrings.
* **Inline Comments**: Use inline comments (\#) to explain complex or non-obvious lines of code. The goal is to explain
the *why*, not just the *what*.

## **7\. Configuration Management**

* **No Hardcoded Values**: Hyperparameters (learning rate, batch size, number of epochs), file paths, and other
configuration settings must not be hardcoded in the training or model scripts.
* **Centralized Config**: All such values will be stored in the src/config.py file. This makes experiments and
modifications easy and safe.

## **8\. Modularity and "Code as Architecture"**

* **Single Responsibility Principle**: Each script and function should have one clear purpose, as outlined in the
project structure.
  * data\_loader.py only handles data.  
  * model.py only defines the network architecture.  
  * train.py orchestrates the training process by importing and using components from the other scripts.  
* **Framework-Specific Practices**: When using PyTorch/TensorFlow, we will follow best practices, such as defining
models as subclasses of torch.nn.Module and explicitly managing device placement (e.g., .to(device)).

## **9\. Logging and Output**

* **Standardized Logging**: The logging module from the Python standard library will be used for logging.  
* **Two Streams**: As defined in the Learning Strategy, we will have two logging streams: a simple, readable status
update printed to the console and a detailed, machine-parsable log saved to /outputs/logs/training.log.
* **Saved Artifacts**: All outputs generated by a run—visualizations, logs, and model weights—must be saved to the
appropriate subdirectory within /outputs/.

## **9.1 Visualization Code Standards**

* **Centralization:** All reusable visualization functions must be implemented in the shared `/plotting` package.
Model-specific visualizations should leverage these utilities whenever possible.
* **Naming:** Visualization functions should use descriptive, action-oriented names (e.g., `plot_learning_curve`,
`plot_confusion_matrix`, `plot_decision_boundary`).
* **Docstrings:** Every visualization function must have a Google-style docstring describing its purpose, arguments,
and expected output (including file paths for saved figures).
* **Reproducibility:** Visualization code that involves randomness (e.g., t-SNE, UMAP) must set random seeds for
reproducibility and document this in the docstring.
* **Saving Figures:** All plots must be saved to the model’s `outputs/visualizations/` directory with clear,
descriptive filenames. Figures should not be shown interactively by default in scripts, but may be displayed in
notebooks.
* **Notebook Integration:** Every model’s analysis notebook must include code cells that generate and display the
required visualizations, with markdown cells interpreting the results. Notebooks should reference the Visualization
Playbooks and Implementation Guide for best practices.
* **Extensibility:** When adding new visualization types, update the Implementation Guide and Playbooks accordingly.

## **10\. Performance and Optimization**

The primary goal is learning, but practical efficiency is important. Our optimization strategy will evolve with the
complexity of the models.

* **Principle**: We will optimize the *scaffolding* around the core model logic (e.g., data loading) but will not use
modern optimizations that would obscure the historical lesson of the model itself.
* **Early Models (NumPy-based)**:  
  * The focus is on **algorithmic clarity**.  
* We will use NumPy's **vectorization** capabilities over Python loops wherever possible, as this is fundamental to
efficient scientific computing in Python.
* We will intentionally **avoid GPU acceleration** at this stage. Experiencing the slower speed of CPU-based training
is part of the historical lesson.
* **Later Models (Framework-based)**:  
* **GPU Acceleration**: All PyTorch/TensorFlow code must be device-agnostic. The training script will detect an
available GPU (like CUDA or MPS) and move the model and data to it automatically. This is a standard practice that
doesn't interfere with learning the model's architecture.
* **Efficient Data Loading**: For larger datasets, we will use the framework's built-in DataLoader classes. We will
leverage their multi-processing capabilities (e.g., setting num\_workers \> 0\) to ensure the data pipeline is not a
bottleneck.
* **Profiling**: If a training run is unexpectedly slow, we will introduce the use of a profiler (e.g.,
torch.profiler) to identify and address the specific bottleneck, treating this as a practical learning exercise in
itself.

## **11\. Open Source Licensing**

To support the project's goal of being a public educational resource, all code within this repository will be licensed
under the **MIT License**. A LICENSE file will be included in the root of the repository, and each README.md file will
include a reference to this license. This permissive license allows anyone to freely use, modify, and distribute the
code for their own learning and projects.
