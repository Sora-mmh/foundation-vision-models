# DeiT (Data-efficient Image Transformers)

## Overview
DeiT is a vision transformer model that achieves competitive results without requiring massive pre-training on large datasets. It uses knowledge distillation to transfer knowledge from a CNN teacher to a transformer student, enabling efficient training.

## Key Features
- **Knowledge Distillation**: Uses both class token and distillation token
- **Efficient Training**: Trains on ImageNet-1k without external data
- **Teacher-Student Framework**: CNN teacher guides transformer student
- **Hard/Soft Distillation**: Supports both hard and soft distillation targets

## Architecture
1. **Vision Transformer Backbone**: Standard ViT architecture
2. **Distillation Token**: Additional token for distillation learning
3. **Teacher Network**: Pre-trained CNN (e.g., RegNet, ResNet)
4. **Dual Heads**: Separate heads for classification and distillation

## Components
- **Patch Embedding**: Splits image into patches
- **Transformer Blocks**: Self-attention and MLP layers
- **Class Token**: For classification task
- **Distillation Token**: For distillation from teacher
- **Dual Prediction Heads**: Separate outputs for student and teacher targets

## Theory
DeiT uses a distillation token that learns from the teacher's predictions, complementing the class token that learns from true labels. This allows the transformer to benefit from both supervised learning and knowledge distillation, achieving better performance with less data.

## Usage
```python
from model import DistilledVisionTransformer, DeiT, DistillationLoss

# Create teacher and student models
teacher = create_teacher_model()
student = DistilledVisionTransformer(img_size=224, patch_size=16, num_classes=1000)

# Create DeiT with distillation
deit = DeiT(teacher_model=teacher, student_model=student)

# Training
student_logits, student_dist_logits, teacher_logits = deit(images)
loss = distillation_loss(student_logits, student_dist_logits, teacher_logits, labels)