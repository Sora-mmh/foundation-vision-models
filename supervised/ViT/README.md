# Vision Transformer (ViT)

## Overview
Vision Transformer (ViT) is a transformer-based model for image classification that treats images as sequences of patches. It applies the standard Transformer architecture directly to images with minimal modifications.

## Key Features
- **Patch-based Processing**: Divides images into fixed-size patches
- **Transformer Architecture**: Uses standard Transformer encoder blocks
- **Position Embeddings**: Adds learnable position embeddings to patches
- **CLS Token**: Uses a special classification token for prediction

## Architecture
1. **Patch Embedding**: Splits image into patches and projects to embeddings
2. **Position Embedding**: Adds positional information to patches
3. **Transformer Encoder**: Multiple layers of self-attention and MLP
4. **Classification Head**: Linear layer for final prediction

## Components
- **Patch Embedding Layer**: Convolutional layer for patch extraction
- **Position Embeddings**: Learnable positional encodings
- **Multi-head Self-Attention**: Captures global dependencies
- **MLP Blocks**: Non-linear transformations
- **Layer Normalization**: Stabilizes training

## Theory
ViT treats image patches as tokens in a sequence, similar to words in NLP. The model learns to attend to relevant patches through self-attention mechanisms. The CLS token aggregates information from all patches for the final classification decision.

## Usage
```python
from model import VisionTransformer

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
logits = vit(images)

# Extract features
features = vit.forward_features(images)