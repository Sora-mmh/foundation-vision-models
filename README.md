# Foundation Models Implementation Suite

A comprehensive collection of state-of-the-art foundation models across various categories, with complete implementations and detailed documentation.

## 📁 Project Structure

```
foundation-models/
├── self-supervised/          # Self-Supervised Learning models
│   ├── MAE/                 # Masked Autoencoder
│   ├── DINO/                # Self-Distillation with No Labels
│   └── SimCLR/              # Simple Framework for Contrastive Learning
├── contrastive-learning/    # Contrastive Learning models
│   ├── CLIP/                # Contrastive Language-Image Pre-training
│   ├── ALIGN/               # Scaling Up Visual and Vision-Language Representation Learning
│   └── MoCo/                # Momentum Contrast
├── knowledge-distillation/  # Knowledge Distillation models
│   ├── DeiT/                # Data-efficient Image Transformers
│   └── TinyViT/             # Compact Vision Transformers
├── masked-modeling/         # Masked Modeling models
│   ├── BEiT/                # BERT Pre-Training of Image Transformers
│   └── SimMIM/              # Simple Masked Image Modeling
├── multimodal/              # Multimodal models
│   ├── BLIP/                # Bootstrapping Language-Image Pre-training
│   ├── Flamingo/            # Visual Language Model
│   └── CoCa/                # Contrastive Captioners
└── supervised-pretraining/  # Supervised Pre-training models
    ├── ViT/                 # Vision Transformer
    ├── Swin/                # Swin Transformer
    └── ConvNeXt/            # Modernized ConvNet
```

## 🗂️ Categories and Models

### 1. Self-Supervised Learning
- **MAE (Masked Autoencoder)**: Asymmetric encoder-decoder architecture that reconstructs masked image patches
- **DINO (Self-Distillation with No Labels)**: Self-distillation framework with momentum teacher
- **SimCLR (Simple Framework for Contrastive Learning)**: Contrastive learning with simple architecture and strong augmentation

### 2. Contrastive Learning
- **CLIP (Contrastive Language-Image Pre-training)**: Learns visual concepts from natural language supervision
- **ALIGN (Scaling Up Visual and Vision-Language Representation Learning)**: Uses billion-scale noisy image-text pairs
- **MoCo (Momentum Contrast)**: Builds dynamic dictionary with queue and moving-averaged encoder

### 3. Knowledge Distillation
- **DeiT (Data-efficient Image Transformers)**: Vision transformer with distillation token for efficient training
- **TinyViT (Compact Vision Transformers)**: Family of compact vision transformers with architectural optimizations

### 4. Masked Modeling
- **BEiT (BERT Pre-Training of Image Transformers)**: BERT-style pre-training for vision transformers
- **SimMIM (Simple Masked Image Modeling)**: Simple framework with direct regression of raw pixel values

### 5. Multimodal
- **BLIP (Bootstrapping Language-Image Pre-training)**: Unified vision-language understanding and generation
- **Flamingo (Visual Language Model)**: Few-shot visual language model with gated cross-attention
- **CoCa (Contrastive Captioners)**: Combines contrastive learning and caption generation

### 6. Supervised Pre-training
- **ViT (Vision Transformer)**: Transformer architecture applied directly to image patches
- **Swin Transformer**: Hierarchical vision transformer with shifted windows
- **ConvNeXt**: Modernized convolutional network competing with vision transformers

## 🚀 Quick Start

### Installation
```bash
# Clone the repository
git clone https://github.com/Sora-mmh/foundation-vision-models.git
cd foundation-models

# Install dependencies
./install.sh
```

### Basic Usage
```python
# Example: Using ViT for image classification
from supervised.ViT.model import VisionTransformer

# Initialize model
vit = VisionTransformer(
    img_size=224,
    patch_size=16,
    num_classes=1000,
    embed_dim=768,
    depth=12,
    num_heads=12
)

# Forward pass
import torch
x = torch.randn(1, 3, 224, 224)
logits = vit(x)
print(f"Output shape: {logits.shape}")
```

### Training Example
```python
# Example: Training MAE
from self_supervised.MAE.model import MaskedAutoencoderViT

# Initialize model
mae = MaskedAutoencoderViT(
    img_size=224,
    patch_size=16,
    in_chans=3,
    embed_dim=768,
    depth=12,
    num_heads=12,
    decoder_embed_dim=512,
    decoder_depth=8,
    decoder_num_heads=16,
    mlp_ratio=4.0,
    mask_ratio=0.75
)

# Training loop
optimizer = torch.optim.AdamW(mae.parameters(), lr=1.5e-4)

for images in dataloader:
    loss, pred, mask = mae(images)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

## 📊 Model Comparison

| Model | Category | Parameters | Key Features |
|-------|----------|------------|--------------|
| ViT | Supervised | 86M | Pure transformer, patch-based processing |
| Swin | Supervised | 29M-197M | Hierarchical, shifted windows |
| ConvNeXt | Supervised | 29M-198M | Modernized CNN, competes with transformers |
| MAE | Self-Supervised | 86M+ | High masking ratio, asymmetric encoder-decoder |
| DINO | Self-Supervised | 22M-86M | Self-distillation, momentum teacher |
| SimCLR | Self-Supervised | varies | Contrastive learning, strong augmentation |
| CLIP | Contrastive | varies | Multimodal, zero-shot capability |
| ALIGN | Contrastive | varies | Billion-scale training, noisy data robust |
| MoCo | Contrastive | varies | Dynamic dictionary, momentum encoder |
| DeiT | Distillation | 22M | Distillation token, teacher-student framework |
| TinyViT | Distillation | 5M-21M | Compact architecture, knowledge distillation |
| BEiT | Masked Modeling | 86M | BERT-style, visual token prediction |
| SimMIM | Masked Modeling | 86M | Simple, direct pixel regression |
| BLIP | Multimodal | varies | Bootstrapping, unified architecture |
| Flamingo | Multimodal | varies | Few-shot learning, gated cross-attention |
| CoCa | Multimodal | varies | Dual objectives, contrastive + captioning |

## 🛠️ Implementation Details

Each model directory contains:
- `model.py`: Complete implementation with architecture definition
- `README.md`: Detailed documentation including:
  - Overview and key features
  - Architecture description
  - Components and theory
  - Usage examples
  - Applications and references

### Common Features:
- **Modular Design**: Easy to understand and modify
- **Complete Implementations**: From scratch implementations
- **Usage Examples**: Ready-to-run code snippets
- **Pre-training Ready**: Configurable for different scales
- **Well Documented**: Detailed explanations and references

## 🎯 Applications

### Computer Vision
- Image classification
- Object detection
- Semantic segmentation
- Image generation

### Multimodal Tasks
- Image-text retrieval
- Visual question answering
- Image captioning
- Cross-modal search

### Transfer Learning
- Feature extraction
- Fine-tuning for downstream tasks
- Few-shot learning
- Zero-shot classification

## 📚 References

Each model includes references to original papers and implementations. Key references include:

1. **ViT**: [An Image is Worth 16x16 Words](https://arxiv.org/abs/2010.11929)
2. **Swin**: [Hierarchical Vision Transformer using Shifted Windows](https://arxiv.org/abs/2103.14030)
3. **ConvNeXt**: [A ConvNet for the 2020s](https://arxiv.org/abs/2201.03545)
4. **MAE**: [Masked Autoencoders Are Scalable Vision Learners](https://arxiv.org/abs/2111.06377)
5. **DINO**: [Emerging Properties in Self-Supervised Vision Transformers](https://arxiv.org/abs/2104.14294)
6. **SimCLR**: [A Simple Framework for Contrastive Learning](https://arxiv.org/abs/2002.05709)
7. **CLIP**: [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020)
8. **ALIGN**: [Scaling Up Visual and Vision-Language Representation Learning](https://arxiv.org/abs/2102.05918)
9. **MoCo**: [Momentum Contrast for Unsupervised Visual Representation Learning](https://arxiv.org/abs/1911.05722)
10. **DeiT**: [Training data-efficient image transformers & distillation through attention](https://arxiv.org/abs/2012.12877)
11. **TinyViT**: [Fast Pretraining Distillation for Small Vision Transformers](https://arxiv.org/abs/2207.10666)
12. **BEiT**: [BERT Pre-Training of Image Transformers](https://arxiv.org/abs/2106.08254)
13. **SimMIM**: [A Simple Framework for Masked Image Modeling](https://arxiv.org/abs/2111.09886)
14. **BLIP**: [Bootstrapping Language-Image Pre-training](https://arxiv.org/abs/2201.12086)
15. **Flamingo**: [A Visual Language Model for Few-Shot Learning](https://arxiv.org/abs/2204.14198)
16. **CoCa**: [Contrastive Captioners are Image-Text Foundation Models](https://arxiv.org/abs/2205.01917)

