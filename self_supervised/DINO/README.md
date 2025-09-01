# DINO (Self-Distillation with No Labels)

## Overview
DINO is a self-supervised learning method that uses self-distillation without labels. It trains a student network to match the output of a teacher network on different views of the same image. The teacher network is updated using an exponential moving average of the student weights.

## Key Features
- **Self-Distillation**: Student network learns from teacher network's predictions
- **Multi-crop Strategy**: Uses global and local crops for different views
- **Momentum Teacher**: Teacher network is updated with exponential moving average
- **Centering**: Prevents collapse by centering the teacher outputs

## Architecture
1. **Backbone Network**: Vision Transformer for feature extraction
2. **Projection Head**: Multi-layer perceptron for projecting features
3. **Student Network**: Network that learns from teacher predictions
4. **Teacher Network**: Momentum-updated version of student network

## Components
- **Vision Transformer**: Standard ViT architecture for feature extraction
- **DINO Head**: Projection head with multiple linear layers
- **Multi-crop Wrapper**: Handles multiple resolution inputs
- **Momentum Update**: EMA update mechanism for teacher network

## Theory
DINO learns representations by matching the output distributions of a student network and a teacher network on different views of the same image. The teacher network provides a target distribution through softmax with low temperature, while the student tries to predict this distribution. The centering mechanism prevents collapse by shifting the teacher outputs away from rare patterns.

## Usage
```python
from model import DINO, VisionTransformer, DINOHead

# Initialize backbones and heads
student = VisionTransformer(img_size=224, patch_size=16, embed_dim=768)
teacher = VisionTransformer(img_size=224, patch_size=16, embed_dim=768)
student_head = DINOHead(768, 65536)
teacher_head = DINOHead(768, 65536)

# Create DINO model
dino = DINO(student, teacher, student_head, teacher_head)

# Training loop
for images in dataloader:
    student_out, teacher_out = dino(images)
    loss, center = dino_loss(student_out, teacher_out, center)
    dino.update_teacher(0.996)
