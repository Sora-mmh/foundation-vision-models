import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import math


class MultiCropWrapper(nn.Module):
    """
    Perform forward pass separately on each resolution input.
    The inputs corresponding to a single resolution are clubbed and single
    forward is run on the same resolution inputs. Hence we do several forward
    passes = number of different resolutions used. We then concatenate all the
    output features and run the head forward on these concatenated features.
    """

    def __init__(self, backbone, head):
        super(MultiCropWrapper, self).__init__()
        # Set modules
        self.backbone = backbone
        self.head = head

    def forward(self, x):
        # Convert to list
        if not isinstance(x, list):
            x = [x]
        idx_crops = torch.cumsum(
            torch.unique_consecutive(
                torch.tensor([inp.shape[-1] for inp in x]),
                return_counts=True,
            )[1],
            0,
        )

        start_idx = 0
        output = []
        for end_idx in idx_crops:
            _out = self.backbone(torch.cat(x[start_idx:end_idx]))
            # If output is a tuple, take the first element
            if isinstance(_out, tuple):
                _out = _out[0]
            output.append(_out)
            start_idx = end_idx

        # Run the head forward on the concatenated features
        return self.head(torch.cat(output))


class DINOHead(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        use_bn=False,
        norm_last_layer=True,
        nlayers=3,
        hidden_dim=2048,
        bottleneck_dim=256,
    ):
        super().__init__()
        nlayers = max(nlayers, 1)
        if nlayers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        else:
            layers = [nn.Linear(in_dim, hidden_dim)]
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)

        self.apply(self._init_weights)
        self.last_layer = nn.utils.weight_norm(
            nn.Linear(bottleneck_dim, out_dim, bias=False)
        )
        self.last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.mlp(x)
        x = F.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        return x


class VisionTransformer(nn.Module):
    """Vision Transformer with support for variable input sizes"""

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        drop_rate=0.0,
    ):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim
        self.patch_size = patch_size

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )

        # Initialize with maximum expected size (for 224x224)
        num_patches = (img_size // patch_size) ** 2
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        # Transformer blocks
        self.blocks = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=embed_dim,
                    nhead=num_heads,
                    dim_feedforward=int(mlp_ratio * embed_dim),
                    dropout=drop_rate,
                    activation=F.gelu,
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)

        # Initialize weights
        torch.nn.init.trunc_normal_(self.pos_embed, std=0.02)
        torch.nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def interpolate_pos_encoding(self, x, original_size=224):
        """Interpolate positional encoding to match the current number of patches"""
        npatch = x.shape[1] - 1  # Current number of patches (excluding cls token)
        N = self.pos_embed.shape[1] - 1  # Original number of positional tokens

        if npatch == N:  # No interpolation needed
            return self.pos_embed

        # Separate class token and patch tokens
        class_pos_embed = self.pos_embed[:, :1]
        patch_pos_embed = self.pos_embed[:, 1:]

        # Calculate grid dimensions
        dim = x.shape[-1]
        orig_grid_size = int(math.sqrt(N))
        current_grid_size = int(math.sqrt(npatch))

        # Reshape to 2D grid and interpolate
        patch_pos_embed = patch_pos_embed.reshape(
            1, orig_grid_size, orig_grid_size, dim
        )
        patch_pos_embed = patch_pos_embed.permute(0, 3, 1, 2)

        # Interpolate
        patch_pos_embed = F.interpolate(
            patch_pos_embed,
            size=(current_grid_size, current_grid_size),
            mode="bicubic",
            align_corners=False,
        )

        # Reshape back to sequence
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).reshape(1, -1, dim)

        # Combine with class token positional encoding
        return torch.cat((class_pos_embed, patch_pos_embed), dim=1)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)  # [B, num_patches, embed_dim]

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)  # [B, num_patches + 1, embed_dim]

        # Interpolate positional encoding
        pos_embed = self.interpolate_pos_encoding(x)
        x = x + pos_embed

        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return x[:, 0]  # Return only cls token


class PatchEmbed(nn.Module):
    """Image to Patch Embedding that handles variable input sizes"""

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2

        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        x = self.proj(x)
        # Dynamically calculate number of patches
        if x.dim() == 4:  # [B, C, H, W]
            B, C, H, W = x.shape
            x = x.flatten(2).transpose(1, 2)  # [B, num_patches, embed_dim]
        return x


class DINO(nn.Module):
    """DINO: Self-Distillation with No Labels"""

    def __init__(self, student, teacher, student_head, teacher_head):
        super().__init__()
        self.student = MultiCropWrapper(student, student_head)
        self.teacher = MultiCropWrapper(teacher, teacher_head)

        # Teacher doesn't need gradients
        for p in self.teacher.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def update_teacher(self, momentum):
        """Momentum update of teacher parameters"""
        for param_s, param_t in zip(
            self.student.backbone.parameters(), self.teacher.backbone.parameters()
        ):
            param_t.data.mul_(momentum).add_((1 - momentum) * param_s.data)

        for param_s, param_t in zip(
            self.student.head.parameters(), self.teacher.head.parameters()
        ):
            param_t.data.mul_(momentum).add_((1 - momentum) * param_s.data)

    def forward(self, x):
        """Forward pass through both student and teacher"""
        # Teacher forward
        with torch.no_grad():
            teacher_output = self.teacher(x)
            teacher_output = F.softmax(teacher_output, dim=-1)

        # Student forward
        student_output = self.student(x)
        student_output = F.log_softmax(student_output, dim=-1)

        return student_output, teacher_output


def dino_loss(student_output, teacher_output, center, temperature=0.1):
    """DINO loss function with centering"""
    # Cross entropy between student and teacher distributions
    loss = -torch.sum(teacher_output * student_output, dim=-1).mean()

    # Update center
    center = center * 0.9 + teacher_output.mean(dim=0) * 0.1

    return loss, center


if __name__ == "__main__":
    # Create student and teacher backbones
    student_backbone = VisionTransformer(
        img_size=224, patch_size=16, embed_dim=768, depth=12, num_heads=12
    )
    teacher_backbone = VisionTransformer(
        img_size=224, patch_size=16, embed_dim=768, depth=12, num_heads=12
    )

    # Create heads
    student_head = DINOHead(768, 65536, nlayers=3, hidden_dim=2048, bottleneck_dim=256)
    teacher_head = DINOHead(768, 65536, nlayers=3, hidden_dim=2048, bottleneck_dim=256)

    # Initialize DINO
    dino = DINO(student_backbone, teacher_backbone, student_head, teacher_head)

    # Example input (global and local crops)
    global_crop = torch.randn(2, 3, 224, 224)
    local_crops = [torch.randn(2, 3, 96, 96) for _ in range(6)]
    all_crops = [global_crop] + local_crops

    # Forward pass
    student_output, teacher_output = dino(all_crops)

    print(f"Student output shape: {student_output.shape}")
    print(f"Teacher output shape: {teacher_output.shape}")

    # Compute loss
    center = torch.zeros(65536)  # Output dimension
    loss, center = dino_loss(student_output, teacher_output, center)

    print(f"Loss: {loss.item():.4f}")

    # Update teacher with momentum
    dino.update_teacher(0.996)
