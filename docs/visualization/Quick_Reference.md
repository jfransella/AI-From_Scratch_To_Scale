# Visualization Quick Reference

This table maps each model archetype to its required visualizations. For details and code examples, see the corresponding section in `docs/visualization/Playbooks.md`.

| Model Type                | Required Visualizations                                      | Playbook Section |
|---------------------------|-------------------------------------------------------------|------------------|
| Perceptron, MLP           | Learning Curves, Confusion Matrix, Decision Boundary        | 1. Classifier    |
| CNN (LeNet, AlexNet, etc) | Learning Curves, Confusion Matrix, Filters, Feature Maps, Grad-CAM, Task Output (BBox/Segmentation) | 2. CNN          |
| RNN, LSTM, Transformer    | Learning Curves, Gradient Flow, Hidden State/Attention Heatmap, Embedding Projection | 3. RNN/Transformer |
| VAE, DCGAN, DDPM          | GAN Loss Curves, Generated Grid, Latent Traversal/Interpolation, Denoising Sequence | 4. Generative    |
| GCN, BitNet               | Learning Curves, Embedding t-SNE/UMAP, Weight Histogram, Loss Landscape | 5. Frontier     |

**How to Use:**
- Find your model type in the left column.
- Implement all required visualizations listed in the middle column.
- For code and best practices, see the Playbooks and Implementation Guide. 