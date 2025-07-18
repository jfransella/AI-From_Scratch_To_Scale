# Project Charter: AI From Scratch to Scale: A Hands-On Journey Through the History of Neural Networks

## 1. Project Goal

The primary goal of this project is twofold: first, to gain a deep, practical, and historical understanding of neural
networks for the primary learner; and second, to create a high-quality, open-source educational resource that allows
others to follow the same journey. This will be achieved by progressively building and analyzing the key neural network
architectures that have defined the field, from its origins to the modern day. The project emphasizes not only *how* the
models work but *why* they were invented and what specific limitations they overcame, making the learning process
transparent and accessible to the community.

## 2. Learner Profile

The project is undertaken by an experienced enterprise architect with a multi-decade background in IT, including
software development, IT operations, and data analytics. The learner is comfortable with technical concepts and desires
a deep, principled understanding of the subject matter, starting from foundational principles.

## 3. Guiding Principles & Tools

To ensure consistency, all work will adhere to the following guidelines:

* **Development Environment**: Models will be built using **Python** within **Cursor**. Initial models will rely on
  libraries like NumPy to build from first principles, progressing to major frameworks as complexity increases.
* **AI Assistance**: **Cursor** will be used as a pair-programmer to accelerate development, explain complex code, and
  generate boilerplate code.
* **Version Control**: **GitHub** will be used to track code changes, manage different model implementations, and
  maintain a history of the project.
* **Open Source Commitment**: This project is intended as a public learning resource. All code will be licensed under a
  permissive open-source license (e.g., MIT License) to encourage sharing, learning, and collaboration. All learning
  content, analysis, and results will be made publicly accessible through the project's GitHub repository.
* **Frameworks**: As models become more complex, we will use industry-standard frameworks, primarily **PyTorch** or
  **TensorFlow**.
* **Scaling Environment**: When local execution becomes impractical due to data or compute requirements, the project
  will move to **Microsoft Azure**, utilizing services like Azure Machine Learning and Notebooks.
* **Data Strategy**: We will start with small, classic datasets (e.g., Iris, MNIST, CIFAR-10) that can be handled
  locally. We will only move to larger datasets (e.g., subsets of ImageNet) when the model's complexity and the learning
  objective require it, which will trigger our move to Azure.
* **Historical Fidelity**: We will stay as close as possible to the original specifications of each model. When modern
  adaptations are used for practicality (e.g., using a framework instead of physically wiring a Perceptron), the
  differences between our implementation and the original will be explicitly explained.
* **Learning Methodology**: For each model, the process will be:

  1. **Understand**: Explain the historical context, the problem it solved, and the core innovation.
  2. **Build**: Implement the model in code.
  3. **Demonstrate Strength**: Train the model on a dataset where it excels.
  4. **Expose Weakness**: Use a different dataset or problem to clearly demonstrate the model's limitations, creating
     the motivation for the next model in our journey.

## 4. Definition of "Build"

The term "build" will evolve as the project progresses:

* **Early Models (e.g., Perceptron, MLP)**: "Build" means implementing the core algorithms and mathematics from scratch
using libraries like NumPy.
* **Intermediate Models (e.g., LeNet-5, ResNet)**: "Build" means using a deep learning framework (like PyTorch) to
assemble the architecture from its constituent layers and implementing the training loop.
* **Modern, Large-Scale Models (e.g., BERT, Diffusion Models)**: "Build" means understanding the architecture in
detail, but the implementation will focus on loading a **pre-trained** version of the model and **fine-tuning** it for a
specific task. Training these models from scratch is beyond the scope of this project.

## 5. Model Engagement Levels

To manage effort and focus on the most critical learning objectives, each model in the roadmap is assigned an
engagement level:

* **Keystone**: These are the most important models that introduced a fundamental paradigm shift. They require the
deepest level of engagement, including a full implementation, training, and the complete "Strength/Weakness" analysis as
defined in our methodology.
* **Conceptual**: These models represent important iterative improvements or alternative ideas. Our engagement will
focus on the "Understand" phase. We will analyze the architecture and innovation in detail, perhaps sketching out
pseudocode, but we will not perform a full build and training run. This allows us to learn the concept efficiently
without redundant implementation.
* **Side-quest**: These models explore a different branch of thought from the main lineage of modern deep learning.
They will be treated like Keystone models (full build and analysis), but we will explicitly frame the work as an
exploration of an alternative, parallel paradigm.

## 6. Project Roadmap: Learning Modules & Models

This roadmap is designed to be a marathon, not a sprint. We will tackle it in logical modules, focusing on the
"keystone" models for full implementation and treating others as conceptual studies where appropriate.

### Module 1: The Foundations (From Scratch)

*The goal here is to understand the basic mechanics of a neuron and a learning algorithm.*

* **1\. The Perceptron** (Keystone) ✅ **COMPLETED**
  * Engine-based implementation with full BaseModel interface
  * Comprehensive experiment suite (strength/weakness analysis)
  * Complete documentation and notebooks
  * Demonstrates linear limitations (XOR ~53% accuracy)
  
* **2\. ADALINE** (Conceptual study of the Delta Rule) 📋 **NEXT**
  * Simple pattern implementation (educational focus)
  * Comparison with Perceptron learning approach
  * Educational analysis of continuous vs. discrete learning
  * Focus on Delta Rule vs. Perceptron Learning Rule
  
* **3\. Multi-Layer Perceptron (MLP)** (Keystone) ✅ **COMPLETED**
  * Simple implementation pattern demonstration
  * XOR breakthrough capability (75% accuracy vs. 53%)
  * Complete visualization and analysis suite
  * Shows power of hidden layers and backpropagation
  
* **4\. Hopfield Network** (Side-quest: understand a different paradigm) 📋 **PLANNED**

### Module 2: The CNN Revolution (Intro to Frameworks)

*The goal is to understand how networks process spatial data like images.*

* **5\. LeNet-5** (Keystone)  
* **6\. AlexNet** (Keystone: focus on ReLU, Dropout, and scale)  
* **7\. VGGNet** (Conceptual: understand "depth")  
* **8\. GoogLeNet** (Conceptual: understand "width" and efficiency)  
* **9\. ResNet** (Keystone: the critical innovation of skip connections)

### Module 3: Applying CNNs to New Problems

*The goal is to see how the CNN building block can be used in complex frameworks.*

* **10\. R-CNN** (Conceptual: understand the original framework)  
* **11\. Faster R-CNN** (Keystone: implement the RPN and end-to-end pipeline)  
* **12\. YOLO** (Keystone: understand a different, single-pass philosophy)  
* **13\. U-Net** (Keystone: focus on segmentation and the skip-connection architecture)  
* **14\. Mask R-CNN** (Conceptual: understand how it extends Faster R-CNN)

### Module 4: The Sequence Models

*The goal is to understand how networks process sequential data like text.*

* **15\. Recurrent Neural Network (RNN)** (Keystone: implement the recurrence from scratch)  
* **16\. LSTM & GRU** (Keystone: implement both to compare the gated mechanisms)  
* **17\. LSTM with Attention** (Keystone: add the attention mechanism to an LSTM)  
* **18\. The Transformer** (Keystone: focus on the self-attention mechanism, abandoning recurrence)

### Module 5: The Generative Era

*The goal is to understand how networks can create new data, not just classify it.*

* **19\. Variational Autoencoder (VAE)** (Keystone)  
* **20\. GAN** (Conceptual: understand the theory)  
* **21\. DCGAN** (Keystone: implement a stable GAN)  
* **22\. Denoising Diffusion Models (DDPM)** (Keystone: implement the core denoising loop)

### Module 6: The Modern Paradigm

*The goal is to understand transfer learning, new data structures, and extreme efficiency.*

* **23\. Graph Convolutional Network (GCN)** (Keystone)  
* **24\. BERT** (Keystone: focus on the **fine-tuning** process, not pre-training)  
* **25\. BitNet 1.58b** (Keystone: focus on Quantization-Aware Training for efficiency)
