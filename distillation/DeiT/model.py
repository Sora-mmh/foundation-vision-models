import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import VisionTransformer
from timm.models.layers import trunc_normal_


class DistilledVisionTransformer(VisionTransformer):
    """Vision Transformer with distillation token"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.dist_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 2, self.embed_dim))
        self.head_dist = (
            nn.Linear(self.embed_dim, self.num_classes)
            if self.num_classes > 0
            else nn.Identity()
        )

        trunc_normal_(self.dist_token, std=0.02)
        trunc_normal_(self.pos_embed, std=0.02)
        self.head_dist.apply(self._init_weights)

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        dist_token = self.dist_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, dist_token, x), dim=1)

        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return x[:, 0], x[:, 1]

    def forward(self, x):
        x, x_dist = self.forward_features(x)
        x = self.head(x)
        x_dist = self.head_dist(x_dist)
        if self.training:
            return x, x_dist
        else:
            # During inference, return the average of classifier and dist head
            return (x + x_dist) / 2


class DeiT(nn.Module):
    """DeiT: Data-efficient Image Transformers with distillation"""

    def __init__(
        self, teacher_model=None, student_model=None, alpha=0.5, temperature=3.0
    ):
        super().__init__()
        self.teacher = teacher_model
        self.student = student_model
        self.alpha = alpha
        self.temperature = temperature

        # Freeze teacher
        if self.teacher is not None:
            for param in self.teacher.parameters():
                param.requires_grad = False

    def forward(self, x):
        if self.training:
            # Student forward
            student_logits, student_dist_logits = self.student(x)

            # Teacher forward
            with torch.no_grad():
                teacher_logits = self.teacher(x)

            return student_logits, student_dist_logits, teacher_logits
        else:
            # Inference
            return self.student(x)


class DistillationLoss(nn.Module):
    """Distillation loss (hard and soft)"""

    def __init__(self, alpha=0.5, temperature=3.0, distillation_type="hard"):
        super().__init__()
        self.alpha = alpha
        self.temperature = temperature
        self.distillation_type = distillation_type
        self.ce_loss = nn.CrossEntropyLoss()
        self.kl_loss = nn.KLDivLoss(reduction="batchmean")

    def forward(self, student_logits, student_dist_logits, teacher_logits, labels):
        # Classification loss
        loss_cls = self.ce_loss(student_logits, labels)

        if self.distillation_type == "hard":
            # Hard distillation: use teacher predictions as targets
            teacher_preds = teacher_logits.argmax(dim=1)
            loss_dist = self.ce_loss(student_dist_logits, teacher_preds)
        else:
            # Soft distillation: use KL divergence with temperature
            soft_teacher = F.softmax(teacher_logits / self.temperature, dim=1)
            soft_student = F.log_softmax(student_dist_logits / self.temperature, dim=1)
            loss_dist = self.kl_loss(soft_student, soft_teacher) * (self.temperature**2)

        # Combined loss
        loss = (1 - self.alpha) * loss_cls + self.alpha * loss_dist
        return loss


if __name__ == "__main__":
    # Create teacher model (typically a CNN like ResNet-50)
    teacher_model = torch.hub.load(
        "pytorch/vision:v0.10.0", "resnet50", pretrained=True
    )
    teacher_model.fc = nn.Linear(
        teacher_model.fc.in_features, 1000
    )  # Adjust for 1000 classes

    # Create student model (DeiT)
    student_model = DistilledVisionTransformer(
        img_size=224,
        patch_size=16,
        in_chans=3,
        num_classes=1000,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        weight_init="",
    )

    # Create DeiT with distillation
    deit = DeiT(
        teacher_model=teacher_model,
        student_model=student_model,
        alpha=0.5,
        temperature=3.0,
    )

    # Example input
    x = torch.randn(4, 3, 224, 224)
    labels = torch.randint(0, 1000, (4,))

    print(f"Input shape: {x.shape}")
    print(f"Labels: {labels}")

    # Forward pass (training)
    deit.train()
    student_logits, student_dist_logits, teacher_logits = deit(x)

    print(f"Student logits shape: {student_logits.shape}")
    print(f"Student dist logits shape: {student_dist_logits.shape}")
    print(f"Teacher logits shape: {teacher_logits.shape}")

    # Compute loss
    loss_fn = DistillationLoss(alpha=0.5, temperature=3.0, distillation_type="soft")
    loss = loss_fn(student_logits, student_dist_logits, teacher_logits, labels)

    print(f"Distillation loss: {loss.item():.4f}")

    # Inference
    deit.eval()
    with torch.no_grad():
        outputs = deit(x)
        print(f"Inference outputs shape: {outputs.shape}")
