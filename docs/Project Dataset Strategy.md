# **Project Dataset Strategy**

This contains a list of all the models we are going to build, which datasets we want the training runs to be able to process, and why. 

### 

### **Module 1: The Foundations 💾**

* **The Perceptron**  
  * **Strength Datasets**:  
    1. Generated AND Gate Data  
    2. Iris Dataset (Setosa vs. Versicolor classes)  
    3. MNIST Dataset (0s vs. 1s only)  
    * **Why**: To demonstrate success on three levels of increasing complexity: a fundamental logic gate, a classic real-world linearly separable problem, and a simple, high-dimensional image classification task.  
  * **Weakness Datasets**:  
    1. Generated XOR Gate Data  
    2. Iris Dataset (Versicolor vs. Virginica classes)  
    3. Generated Concentric Circles or Moons Data  
    * **Why**: To show its failure on a canonical non-linear problem (XOR), a real-world non-linear dataset, and a visually complex non-linear pattern, which motivates the need for the MLP.  
* **ADALINE**  
  * **Strength Datasets**:  
    1. Generated 2D Linearly Separable Points  
    2. Iris Dataset (Setosa vs. Versicolor classes)  
    3. MNIST Dataset (0s vs. 1s only)  
    * **Why**: To visualize the smooth, **gradient-based convergence** of the Delta Rule on problems of increasing complexity, from a simple 2D plot to a real-world dataset and finally to high-dimensional image data.  
  * **Weakness Datasets**:  
    1. Generated XOR Gate Data  
    2. Iris Dataset (Versicolor vs. Virginica classes)  
    * **Why**: To reinforce that single-layer networks, regardless of using a more sophisticated learning rule, **cannot solve non-linearly separable problems**.  
* **Multi-Layer Perceptron (MLP)**  
  * **Strength Datasets**:  
    1. Generated XOR Gate Data  
    2. Generated Concentric Circles or Moons Data  
    3. Iris Dataset (Versicolor vs. Virginica classes)  
    4. MNIST Digits (full dataset)  
    * **Why**: To show it triumphantly solves the Perceptron's failure cases (canonical, visual, and real-world), and can handle a high-dimensional, multi-class classification problem.  
  * **Weakness Dataset**: CIFAR-10.  
    * **Why**: To expose its **lack of spatial awareness**. Its poor performance on these more complex color images will motivate the need for CNNs.  
* **Hopfield Network**  
  * **Strength Datasets**:  
    1. Generated Small Binary Images (e.g., 5x5 letters like 'X', 'O', 'T').  
    2. Binarized MNIST digits (distinct digits like '0', '1', '7').  
    3. A single stored pattern (from either set) with increasing levels of noise.  
    * **Why**: To demonstrate its function as an **auto-associative memory** on a simple generated case, a more complex real-world case (MNIST), and to explicitly test its robustness to noise.  
  * **Weakness Datasets**:  
    1. Storing a set of similar, non-orthogonal generated letters (e.g., 'P' and 'F').  
    2. Attempting to store all 10 binarized MNIST digits.  
    3. Storing a set of similar, non-orthogonal digits (e.g., '3' and '8').  
    * **Why**: To show the two primary failure modes—**spurious states** from correlated memories and **exceeding storage capacity**—first on a simple generated set and then on a more complex, real-world dataset.

### **Module 2: The CNN Revolution 🖼️**

* **LeNet-5**  
  * **Strength Dataset**: MNIST Handwritten Digits.  
    * **Why**: This is the **historically perfect dataset** that LeNet-5 was designed for, demonstrating the power of basic convolutions.  
  * **Weakness Dataset**: CIFAR-10.  
    * **Why**: To show its simple architecture **struggles with more complex, real-world color images**, motivating the need for a deeper model like AlexNet.  
* **AlexNet**  
  * **Strength Dataset**: CIFAR-10 & ImageNette (a subset of ImageNet).  
    * **Why**: To show its deeper architecture and modern features (ReLU, Dropout) **decisively beat LeNet-5** on a harder dataset and can handle ImageNet-style data.  
  * **Weakness Dataset**: Architectural Analysis.  
    * **Why**: To discuss its **ad-hoc, unprincipled design**, which raises the question of how to scale networks methodically, leading to VGGNet and GoogLeNet.  
* **ResNet**  
  * **Strength Dataset**: CIFAR-10 / CIFAR-100.  
    * **Why**: To run a controlled experiment showing a plain deep network fails while a ResNet **solves the degradation problem**, proving the value of **skip connections**.  
  * **Weakness Dataset**: Problem-type Analysis.  
    * **Why**: To expose it as an **encoder-only architecture** excellent for classification but ill-suited for tasks requiring a spatially rich output, like segmentation. This motivates U-Net.

### **Module 3: Applying CNNs 🎯**

* **Faster R-CNN**  
  * **Strength Dataset**: The Oxford-IIIT Pet Dataset.  
    * **Why**: To implement a powerful, **accurate two-stage object detector** on a manageable, real-world dataset with clean bounding box annotations.  
  * **Weakness Dataset**: Inference Time Analysis.  
    * **Why**: To show that its two-stage nature, while accurate, is **not fast enough for real-time applications**, motivating the single-stage approach of YOLO.  
* **YOLO**  
  * **Strength Dataset**: The Oxford-IIIT Pet Dataset.  
    * **Why**: To directly compare it to Faster R-CNN on the same data, highlighting its primary advantage: **real-time inference speed**.  
  * **Weakness Dataset**: Images with Small/Crowded Objects.  
    * **Why**: To expose its weakness in **detecting small or overlapping objects**, showcasing the trade-off it makes between speed and precision.  
* **U-Net**  
  * **Strength Dataset**: The Oxford-IIIT Pet Dataset (using segmentation masks).  
    * **Why**: To show its elegant encoder-decoder architecture can produce **precise pixel-level semantic segmentations**, solving the "encoder-only" problem from Module 2\.  
  * **Weakness Dataset**: Images with multiple instances of the same class.  
    * **Why**: To show it can't distinguish between individual instances (e.g., two dogs become one "dog" blob), exposing the need for **instance segmentation**.

### **Module 4: The Sequence Models ✍️**

* **Recurrent Neural Network (RNN)**  
  * **Strength Dataset**: Character-level text (e.g., Shakespeare).  
    * **Why**: To demonstrate its ability to model sequences by **generating coherent text** over short spans.  
  * **Weakness Dataset**: Generated "Long-Term Copy" Task.  
    * **Why**: To empirically prove it **cannot learn long-term dependencies** due to the vanishing gradient problem, motivating LSTMs.  
* **LSTM & GRU**  
  * **Strength Dataset**: IMDb Movie Reviews.  
    * **Why**: To show their **gating mechanisms** can capture long-range context to perform sentiment analysis on entire reviews.  
  * **Weakness Dataset**: Small English-to-French Translation Subset.  
    * **Why**: To expose the **fixed-vector bottleneck** of the encoder-decoder model, where translation quality fails on long sentences, motivating attention.  
* **LSTM with Attention**  
  * **Strength Dataset**: Small English-to-French Translation Subset.  
    * **Why**: To show a marked improvement in translation and, crucially, to **visualize the attention weights** to see how the model intelligently focuses on different input words.  
  * **Weakness Dataset**: Architectural Analysis.  
    * **Why**: To identify its **inherent sequential nature** as a computational bottleneck that prevents parallelization, motivating the Transformer.  
* **The Transformer**  
  * **Strength Dataset**: Small English-to-French Translation Subset & IMDb Reviews.  
    * **Why**: To show its **self-attention mechanism** can achieve state-of-the-art results without recurrence and can be used as a powerful general text classifier.  
  * **Weakness Dataset**: Complexity Analysis.  
    * **Why**: To discuss its primary weakness: the **quadratic complexity (**O(n2)**) of self-attention**, which makes it very expensive for extremely long sequences.

### **Module 5: The Generative Era ✨**

* **Variational Autoencoder (VAE)**  
  * **Strength Dataset**: Fashion-MNIST.  
    * **Why**: To visualize the learned **probabilistic latent space** and generate novel images by sampling from it.  
  * **Weakness Dataset**: Analysis of generated samples.  
    * **Why**: To show its primary weakness: the tendency to produce **blurry, overly smooth images**, which motivates GANs.  
* **DCGAN**  
  * **Strength Dataset**: CelebA (celebrity faces).  
    * **Why**: To demonstrate a **stable GAN training process** that produces sharp, plausible images, solving the weaknesses of both VAEs and original GANs.  
  * **Weakness Dataset**: Analysis of the generative process.  
    * **Why**: To highlight its **lack of control** over the generated output, motivating more controllable generative models like DDPMs.  
* **Denoising Diffusion Models (DDPM)**  
  * **Strength Dataset**: CIFAR-10.  
    * **Why**: To implement the **iterative denoising process** and generate images of significantly higher quality and diversity than the GAN.  
  * **Weakness Dataset**: Inference Time Analysis.  
    * **Why**: To expose its primary drawback: a very **slow sampling speed** due to its sequential, iterative nature.

### **Module 6: The Modern Paradigm 🚀**

* **Graph Convolutional Network (GCN)**  
  * **Strength Dataset**: Cora Citation Network.  
    * **Why**: To show it can **leverage graph structure** to dramatically outperform an MLP on a node classification task.  
  * **Weakness Dataset**: Architectural Analysis.  
    * **Why**: To discuss its limitation as a **transductive model**, which cannot easily generalize to new, unseen nodes.  
* **BERT**  
  * **Strength Dataset**: IMDb Movie Reviews.  
    * **Why**: To demonstrate the immense power of **transfer learning** by quickly fine-tuning a massive pre-trained model to achieve state-of-the-art performance.  
  * **Weakness Dataset**: Model Size Analysis.  
    * **Why**: To highlight its **enormous computational cost and size** as a major barrier to deployment, motivating the need for efficiency.  
* **BitNet 1.58b**  
  * **Strength Dataset**: IMDb fine-tuning task.  
    * **Why**: To demonstrate the massive **efficiency gains** (memory, speed) of 1.58-bit quantization by a full-precision model.  
  * **Weakness Dataset**: Accuracy vs. Efficiency Analysis.  
    * **Why**: To analyze the fundamental **trade-off between extreme efficiency and a potential drop in accuracy**, a key challenge in modern AI.