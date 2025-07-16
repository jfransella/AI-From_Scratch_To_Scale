# Visualization Playbooks

This document provides a concise reference for required visualizations for each major model archetype in the project. For each model type, it lists the core plots, their purpose, and example code snippets for generating them using the shared `/plotting` package. For detailed rationale and interpretive guidance, see `docs/strategy/Visualization_Ideas.md`.

---

## 1. Foundational Classifier Playbook (Perceptron, MLP)

**Required Visualizations:**

- Learning Curves (loss & accuracy)
- Confusion Matrix
- Decision Boundary Plot (for 2D data)

**Example Usage:**

```python
from plotting import plot_learning_curve, plot_confusion_matrix, plot_decision_boundary

# After training
plot_learning_curve(train_losses, val_losses, save_path="outputs/visualizations/learning_curve.png")
plot_confusion_matrix(y_true, y_pred, save_path="outputs/visualizations/confusion_matrix.png")
plot_decision_boundary(model, X, y, save_path="outputs/visualizations/decision_boundary.png")
```

---

## 2. CNN Playbook (LeNet-5, AlexNet, ResNet, YOLO, U-Net)

**Required Visualizations:**

- Learning Curves
- Confusion Matrix
- First-Layer Filters
- Feature Maps
- Grad-CAM
- Application-Specific Output (Bounding Box or Segmentation Mask)

**Example Usage:**

```python
from plotting import plot_first_layer_filters, plot_feature_maps, plot_grad_cam

plot_first_layer_filters(model, save_path="outputs/visualizations/filters.png")
plot_feature_maps(model, sample_image, save_path="outputs/visualizations/feature_maps.png")
plot_grad_cam(model, sample_image, target_class, save_path="outputs/visualizations/grad_cam.png")
```

---

## 3. Recurrent/Transformer Playbook (RNN, LSTM, Transformer, BERT)

**Required Visualizations:**

- Learning Curves
- Gradient Flow (for RNNs)
- Hidden State Heatmap (RNNs)
- Attention Heatmap/BertViz (Transformers)
- t-SNE/UMAP Embedding Projection

**Example Usage:**

```python
from plotting import plot_gradient_flow, plot_hidden_state_heatmap, plot_attention_heatmap, plot_embedding_projection

plot_gradient_flow(model, save_path="outputs/visualizations/gradient_flow.png")
plot_hidden_state_heatmap(hidden_states, save_path="outputs/visualizations/hidden_state_heatmap.png")
plot_attention_heatmap(attention_weights, save_path="outputs/visualizations/attention_heatmap.png")
plot_embedding_projection(embeddings, labels, save_path="outputs/visualizations/embedding_projection.png")
```

---

## 4. Generative Playbook (VAE, DCGAN, DDPM)

**Required Visualizations:**

- Generator/Discriminator Loss Curves
- Grid of Generated Samples
- Latent Space Traversal/Interpolation
- Denoising Process Sequence (for DDPM)

**Example Usage:**

```python
from plotting import plot_gan_losses, plot_generated_grid, plot_latent_traversal, plot_denoising_process

plot_gan_losses(gen_losses, disc_losses, save_path="outputs/visualizations/gan_losses.png")
plot_generated_grid(generator, fixed_noise, save_path="outputs/visualizations/generated_grid.png")
plot_latent_traversal(vae, save_path="outputs/visualizations/latent_traversal.png")
plot_denoising_process(ddpm, save_path="outputs/visualizations/denoising_process.gif")
```

---

## 5. Frontier Playbook (GCN, BitNet)

**Required Visualizations:**

- Learning Curves
- t-SNE/UMAP of Node Embeddings (GCN)
- Weight Histogram (BitNet)
- Loss Landscape Visualization

**Example Usage:**

```python
from plotting import plot_weight_histogram, plot_loss_landscape

plot_weight_histogram(model, save_path="outputs/visualizations/weight_histogram.png")
plot_loss_landscape(model, save_path="outputs/visualizations/loss_landscape.png")
```

---

**For more details and interpretive guidance, see the Implementation Guide and `docs/strategy/Visualization_Ideas.md`.**
