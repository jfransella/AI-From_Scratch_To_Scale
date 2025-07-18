# **Complete Perceptron Example Implementation**

This directory contains a complete, working implementation of the Perceptron model that demonstrates all the patterns,
templates, and best practices established in the "AI From Scratch to Scale" project.

## **Purpose**

This example serves as a reference implementation that shows:

- How to use all the templates effectively
- Proper integration with shared infrastructure
- Best practices for model implementation
- Complete workflow from setup to analysis

## **What's Included**

### **Implementation Files**

- `src/constants.py` - Model metadata and fixed values
- `src/config.py` - Configuration management with experiments
- `src/model.py` - Perceptron model implementation
- `src/train.py` - Training script with proper argument parsing
- `src/evaluate.py` - Evaluation script with visualization
- `requirements.txt` - Model-specific dependencies

### **Documentation**

- `notebooks/01_Theory_and_Intuition.ipynb` - Historical context and mathematical intuition
- `notebooks/02_Code_Walkthrough.ipynb` - Code explanation and demonstration
- `notebooks/03_Empirical_Analysis.ipynb` - Results analysis and conclusions

### **Testing**

- `tests/test_perceptron.py` - Unit tests for the model
- `tests/test_config.py` - Configuration validation tests
- `tests/test_integration.py` - Integration tests

## **Key Features Demonstrated**

### **1. Template Usage**

- Uses all provided templates as starting points
- Shows how to customize templates for specific models
- Demonstrates proper error handling and validation

### **2. Configuration Management**

- Implements hierarchical configuration system
- Shows experiment-specific parameter overrides
- Demonstrates environment-specific configurations

### **3. Shared Infrastructure Integration**

- Properly integrates with `data_utils` for dataset loading
- Uses `engine` for training and evaluation
- Leverages `plotting` for visualization generation
- Utilizes `utils` for logging and reproducibility

### **4. Best Practices**

- Follows all coding standards and naming conventions
- Implements proper error handling and logging
- Uses appropriate testing strategies
- Maintains clear documentation

## **Quick Start**

```powershell
# Navigate to example directory
Set-Location docs\examples\01_perceptron_complete_example

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -r ..\..\..\requirements-dev.txt
pip install -e ..\..\..

# Run quick test
python src\train.py --experiment debug_small --epochs 2 --debug

# Train on strength dataset (AND gate)
python src\train.py --experiment and_gate --visualize

# Train on weakness dataset (XOR gate)
python src\train.py --experiment xor --visualize

# Evaluate trained model
python src\evaluate.py --checkpoint outputs\models\and_gate_model.pth --experiment and_gate --visualize
```text`n## **Learning Objectives**

By studying this example, you will understand:

1. **How to implement a complete model** from templates to working code
2. **How to integrate with shared infrastructure** effectively
3. **How to structure experiments** for educational value
4. **How to document and test** your implementation
5. **How to follow project conventions** consistently

## **Expected Results**

### **Strength Experiments**

- **AND Gate**: Should achieve 100% accuracy quickly
- **Iris Easy**: Should separate Setosa from others perfectly
- **MNIST Binary**: Should achieve reasonable accuracy on simple digits

### **Weakness Experiments**

- **XOR Gate**: Should fail to learn (demonstrates linear limitation)
- **Iris Hard**: Should struggle with non-linear separation
- **Circles**: Should fail on non-linearly separable data

## **File Structure**

```text`ndocs\examples\01_perceptron_complete_example\
├── README.md                           # This file
├── src\                               # Source code
│   ├── __init__.py
│   ├── constants.py                   # Model metadata
│   ├── config.py                      # Configuration management
│   ├── model.py                       # Perceptron implementation
│   ├── train.py                       # Training script
│   └── evaluate.py                    # Evaluation script
├── notebooks\                         # Analysis notebooks
│   ├── 01_Theory_and_Intuition.ipynb
│   ├── 02_Code_Walkthrough.ipynb
│   └── 03_Empirical_Analysis.ipynb
├── tests\                             # Unit tests
│   ├── test_perceptron.py
│   ├── test_config.py
│   └── test_integration.py
├── outputs\                           # Generated files
│   ├── logs\
│   ├── models\
│   └── visualizations\
├── .venv\                            # Virtual environment
├── requirements.txt                   # Dependencies
└── setup_instructions.md             # Detailed setup guide
```text`n## **Using as Reference**

This example is designed to be:

- **Copied and modified** for new model implementations
- **Studied** to understand best practices
- **Referenced** when implementing similar patterns
- **Extended** with additional features

## **Next Steps**

After studying this example:

1. Try implementing a similar model (ADALINE) using the same patterns
2. Experiment with different configurations and datasets
3. Add your own visualizations and analysis
4. Extend the model with additional features

## **Common Adaptations**

When adapting this example for other models:

### **For Linear Models (ADALINE)**

- Change activation function in `model.py`
- Update learning rule in training loop
- Modify constants for historical accuracy

### **For Non-Linear Models (MLP)**

- Add hidden layers in `model.py`
- Update configuration for layer parameters
- Add feature visualization capabilities

### **For Specialized Models (CNN, RNN)**

- Extend configuration for domain-specific parameters
- Add specialized data preprocessing
- Include domain-specific visualizations

## **Support**

If you encounter issues with this example:

1. Check the [Development FAQ](../../Development_FAQ.md)
2. Review the [AI Development Guide](../../AI_Development_Guide.md)
3. Ensure all dependencies are installed correctly
4. Verify virtual environment is activated

This example represents the gold standard for model implementation in this project. Use it as your reference for
creating high-quality, educational, and maintainable code.