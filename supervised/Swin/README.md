# Swin Transformer

## Overview
Swin Transformer is a hierarchical vision transformer that computes representations with shifted windows. This hierarchical architecture brings efficiency and makes it suitable for a wide range of vision tasks.

## Key Features
- **Hierarchical Architecture**: Builds feature maps at multiple scales
- **Shifted Windows**: Enables cross-window connections while maintaining efficiency
- **Linear Computational Complexity**: Computationally efficient relative to image size
- **Versatile Backbone**: Suitable for various vision tasks

## Architecture
1. **Patch Partition**: Splits image into non-overlapping patches
2. **Linear Embedding**: Projects patches to arbitrary dimension
3. **Swin Transformer Blocks**: Self-attention with shifted windows
4. **Patch Merging**: Reduces resolution and increases dimension
5. **Classification Head**: Global average pooling and linear projection

## Components
- **Window-based Self-Attention**: Computes attention within local windows
- **Shifted Window Partitioning**: Enables cross-window connections
- **Patch Merging Layers**: Reduce spatial resolution while increasing channels
- **Relative Position Bias**: Incorporates positional information

## Theory
Swin Transformer uses a hierarchical structure where self-attention is computed within non-overlapping local windows. The shifted window approach allows information exchange between windows in consecutive layers while maintaining linear computational complexity with respect to image size.

## Usage
```python
from model import SwinTransformer

# Initialize model
swin = SwinTransformer(
    img_size=224,
    patch_size=4,
    num_classes=1000,
    embed_dim=96,
    depths=[2, 2, 6, 2],
    num_heads=[3, 6, 12, 24],
    window_size=7
)

# Forward pass
logits = swin(images)

# Extract features
features = swin.forward_features(images)