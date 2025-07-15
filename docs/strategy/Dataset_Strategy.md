# **Project Dataset Strategy**

This document contains the experimental design for all 25 neural network models, specifying which datasets demonstrate each model's strengths and weaknesses, plus the technical implementation standards for data preprocessing, loading, and management.

## **Table of Contents**
1. [Dataset Categories & Technical Requirements](#dataset-categories--technical-requirements)
2. [Data Preprocessing Standards](#data-preprocessing-standards)
3. [Dataset Loading Infrastructure](#dataset-loading-infrastructure)
4. [File Organization & Storage](#file-organization--storage)
5. [Performance & Optimization](#performance--optimization)
6. [Experimental Design by Model](#experimental-design-by-model)
7. [Data Validation & Quality Checks](#data-validation--quality-checks)

---

## **Dataset Categories & Technical Requirements**

### **1. Synthetic/Generated Data**
- **Examples**: XOR, AND gates, concentric circles, moons, binary patterns
- **Storage**: JSON/CSV for structured data, NPZ for arrays
- **Preprocessing**: Minimal - normalize to [0,1] or [-1,1] range
- **Loading**: Generate on-demand or cache in `data\generated\`
- **Validation**: Verify mathematical properties (e.g., XOR non-linearity)

### **2. Classic ML Datasets**
- **Examples**: Iris, Wine, Boston Housing
- **Storage**: CSV with metadata JSON
- **Preprocessing**: Standard normalization, categorical encoding
- **Loading**: Scikit-learn compatibility layer
- **Validation**: Cross-reference with sklearn versions

### **3. Image Classification**
- **Examples**: MNIST, CIFAR-10/100, Fashion-MNIST, ImageNet subsets
- **Storage**: HDF5 for large datasets, PNG/JPG for inspection
- **Preprocessing**: Resize, normalize, augmentation pipelines
- **Loading**: Batch-optimized with DataLoader
- **Validation**: Shape consistency, pixel range verification

### **4. Object Detection**
- **Examples**: Oxford-IIIT Pet Dataset, COCO subsets
- **Storage**: Images + JSON annotations (COCO format)
- **Preprocessing**: Bounding box normalization, augmentation
- **Loading**: Custom collate functions for variable batch sizes
- **Validation**: Box coordinate validity, class label consistency

### **5. Sequence Data**
- **Examples**: Shakespeare text, IMDb reviews, translation pairs
- **Storage**: TXT files with tokenizer metadata
- **Preprocessing**: Tokenization, padding, sequence length management
- **Loading**: Bucket sampling for efficient batching
- **Validation**: Vocabulary consistency, sequence length distribution

### **6. Graph Data**
- **Examples**: Cora citation network, social networks
- **Storage**: Adjacency matrices + node features (NPZ/HDF5)
- **Preprocessing**: Graph normalization, feature scaling
- **Loading**: Sparse tensor handling, neighbor sampling
- **Validation**: Graph connectivity, feature matrix alignment

---

## **Data Preprocessing Standards**

### **Image Preprocessing Pipeline**
```python
# Standard transformations by dataset type
MNIST_TRANSFORMS = {
    'train': [ToTensor(), Normalize((0.1307,), (0.3081,))],
    'test': [ToTensor(), Normalize((0.1307,), (0.3081,))]
}

CIFAR10_TRANSFORMS = {
    'train': [
        RandomHorizontalFlip(p=0.5),
        RandomCrop(32, padding=4),
        ToTensor(),
        Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ],
    'test': [
        ToTensor(),
        Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ]
}
```

### **Text Preprocessing Standards**
```python
# Tokenization and encoding patterns
TEXT_PREPROCESSING = {
    'character_level': {
        'vocab_size': 256,  # Extended ASCII
        'sequence_length': 100,
        'padding': 'post'
    },
    'word_level': {
        'vocab_size': 10000,
        'sequence_length': 512,
        'padding': 'post',
        'oov_token': '<UNK>'
    }
}
```

### **Normalization Standards**
- **Images**: [0, 1] range with dataset-specific mean/std
- **Tabular**: StandardScaler for continuous, OneHot for categorical
- **Text**: Vocabulary-based integer encoding
- **Graph**: Row-normalized adjacency matrices

---

## **Dataset Loading Infrastructure**

### **Base Dataset Class Structure**
```python
class BaseDataset:
    def __init__(self, data_dir, split='train', transform=None):
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        self.data = self._load_data()
        
    def _load_data(self):
        """Load data from storage - implement in subclasses"""
        raise NotImplementedError
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        sample = self.data[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample
```

### **DataLoader Configuration Standards**
```python
DATALOADER_CONFIGS = {
    'small_datasets': {  # < 10K samples
        'batch_size': 64,
        'shuffle': True,
        'num_workers': 2,
        'pin_memory': True
    },
    'medium_datasets': {  # 10K - 100K samples
        'batch_size': 128,
        'shuffle': True,
        'num_workers': 4,
        'pin_memory': True
    },
    'large_datasets': {  # > 100K samples
        'batch_size': 256,
        'shuffle': True,
        'num_workers': 8,
        'pin_memory': True,
        'persistent_workers': True
    }
}
```

### **Caching Strategy**
- **Generated Data**: Cache in `data\cache\` with timestamp validation
- **Processed Data**: HDF5 files for large datasets, pickle for small
- **Intermediate Results**: Use joblib for sklearn-compatible caching
- **Cache Invalidation**: Version-based cache keys

---

## **File Organization & Storage**

### **Directory Structure**
```
data\
├── raw\                    # Original, unprocessed data
│   ├── cifar10\
│   ├── mnist\
│   └── text\
├── processed\              # Preprocessed, ready-to-use data
│   ├── cifar10\
│   │   ├── train.h5
│   │   ├── test.h5
│   │   └── metadata.json
│   └── mnist\
├── generated\              # Synthetic datasets
│   ├── xor_data.npz
│   ├── circles_data.npz
│   └── generation_params.json
├── cache\                  # Temporary cached data
├── external\               # External dataset downloads
└── validation\             # Dataset validation reports
```

### **File Naming Conventions**
- **Raw Data**: `{dataset_name}_raw.{ext}`
- **Processed Data**: `{dataset_name}_{split}.{ext}`
- **Generated Data**: `{pattern_type}_{params_hash}.npz`
- **Metadata**: `{dataset_name}_metadata.json`

### **Metadata Schema**
```json
{
    "dataset_name": "cifar10",
    "version": "1.0.0",
    "splits": {
        "train": {"samples": 50000, "path": "train.h5"},
        "test": {"samples": 10000, "path": "test.h5"}
    },
    "preprocessing": {
        "normalization": "imagenet_stats",
        "augmentation": ["horizontal_flip", "random_crop"]
    },
    "created_at": "2024-01-01T00:00:00Z",
    "checksum": "abc123def456"
}
```

---

## **Performance & Optimization**

### **Memory Management**
- **Large Datasets**: Use HDF5 with chunking for partial loading
- **Batch Processing**: Implement lazy loading with generators
- **Memory Monitoring**: Track peak memory usage during loading

### **I/O Optimization**
- **SSD Storage**: Store frequently accessed data on fastest available storage
- **Prefetching**: Use DataLoader prefetch_factor for pipeline optimization
- **Compression**: Use LZ4 compression for intermediate files

### **Parallel Processing**
- **Data Generation**: Use joblib for embarrassingly parallel tasks
- **Preprocessing**: Multiprocessing for CPU-intensive transformations
- **Loading**: Optimized num_workers based on dataset size

---

## **Experimental Design by Model**

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

---

## **Data Validation & Quality Checks**

### **Automated Validation Pipeline**
```python
def validate_dataset(dataset_path, expected_schema):
    """Comprehensive dataset validation"""
    checks = {
        'file_integrity': verify_file_checksums(dataset_path),
        'schema_compliance': validate_schema(dataset_path, expected_schema),
        'data_quality': check_data_quality(dataset_path),
        'split_consistency': verify_split_integrity(dataset_path),
        'preprocessing_validity': validate_preprocessing(dataset_path)
    }
    return checks
```

### **Quality Metrics**
- **Completeness**: No missing values in required fields
- **Consistency**: Uniform data types and ranges across splits
- **Accuracy**: Correctness of labels and annotations
- **Timeliness**: Data freshness and version tracking

### **Validation Reports**
- **Daily**: Automated checks on frequently used datasets
- **Pre-training**: Validation before model training begins
- **Post-processing**: Verification after preprocessing steps
- **Manual**: Periodic human inspection of sample data

### **Error Handling**
- **Graceful Degradation**: Fallback to cached versions on corruption
- **Automatic Retry**: Retry failed downloads with exponential backoff
- **Data Recovery**: Strategies for handling partial data loss
- **Alerting**: Notifications for critical data quality issues

---

## **Implementation Examples**

### **Loading CIFAR-10 with Proper Preprocessing**
```python
def load_cifar10(data_dir, split='train', batch_size=128):
    """Standard CIFAR-10 loading with proper preprocessing"""
    transform = CIFAR10_TRANSFORMS[split]
    dataset = CIFAR10Dataset(data_dir, split=split, transform=transform)
    
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size,
        shuffle=(split == 'train'),
        num_workers=4,
        pin_memory=True
    )
    
    return dataloader
```

### **Generating XOR Dataset**
```python
def generate_xor_data(n_samples=1000, noise_level=0.1):
    """Generate XOR dataset with controlled noise"""
    X = np.random.rand(n_samples, 2)
    y = (X[:, 0] > 0.5) ^ (X[:, 1] > 0.5)  # XOR logic
    
    # Add noise
    X += np.random.normal(0, noise_level, X.shape)
    
    return X.astype(np.float32), y.astype(np.int64)
```

### **Text Dataset with Tokenization**
```python
def prepare_text_dataset(text_file, vocab_size=10000, seq_length=512):
    """Prepare text dataset with proper tokenization"""
    with open(text_file, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Tokenization
    tokenizer = build_tokenizer(text, vocab_size=vocab_size)
    
    # Sequence creation
    sequences = create_sequences(tokenizer.encode(text), seq_length)
    
    return sequences, tokenizer
```

---

## **Best Practices Summary**

1. **Always validate data** before training begins
2. **Use appropriate batch sizes** based on dataset size and memory constraints
3. **Implement proper error handling** for data loading failures
4. **Cache preprocessed data** to avoid repeated computation
5. **Monitor memory usage** during data loading and preprocessing
6. **Use version control** for datasets and preprocessing pipelines
7. **Document all preprocessing steps** for reproducibility
8. **Test data loading** on small samples before full dataset processing
9. **Use appropriate data types** (float32 vs float64) for memory efficiency
10. **Implement dataset-specific validation** beyond generic checks