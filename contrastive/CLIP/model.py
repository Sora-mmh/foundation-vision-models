import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import open_clip


class CLIPModel(nn.Module):
    """CLIP model with image and text encoders"""

    def __init__(self, model_name="ViT-B/32", pretrained="openai"):
        super().__init__()
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.tokenizer = open_clip.get_tokenizer(model_name)
        if pretrained:
            for param in self.model.parameters():
                param.requires_grad = False

    def encode_image(self, images):
        return self.model.encode_image(images)

    def encode_text(self, texts):
        return self.model.encode_text(texts)

    def forward(self, images, texts):
        image_features = self.encode_image(images)
        text_features = self.encode_text(texts)
        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)
        return image_features, text_features


class CLIPLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.logit_scale = nn.Parameter(
            torch.ones([]) * torch.tensor(1.0 / temperature).log()
        )

    def forward(self, image_features, text_features):
        # Compute similarity matrix
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logit_scale * text_features @ image_features.t()
        # Create labels
        batch_size = image_features.shape[0]
        labels = torch.arange(batch_size, device=image_features.device)
        # Compute loss
        loss_i = F.cross_entropy(logits_per_image, labels)
        loss_t = F.cross_entropy(logits_per_text, labels)
        loss = (loss_i + loss_t) / 2
        return loss


if __name__ == "__main__":
    clip = CLIPModel("ViT-B/32", pretrained="openai")
    images = torch.randn(4, 3, 224, 224)
    texts = [
        "a photo of a cat",
        "a picture of a dog",
        "an image of a bird",
        "a photograph of a car",
    ]
    tokenizer = clip.tokenizer
    text_tokens = tokenizer(texts)
    print(f"Image shape: {images.shape}")
    print(f"Text tokens shape: {text_tokens.shape}")
    image_features, text_features = clip(images, text_tokens)
    print(f"Image features shape: {image_features.shape}")
    print(f"Text features shape: {text_features.shape}")
    # Compute similarity
    similarity = image_features @ text_features.t()
    print(f"Similarity matrix:\n{similarity}")
    # Compute loss
    loss_fn = CLIPLoss()
    loss = loss_fn(image_features, text_features)
    print(f"CLIP loss: {loss.item():.4f}")

    print("Zero-shot classification example")
    with torch.no_grad():
        # Class names
        class_names = ["cat", "dog", "bird", "car", "truck"]
        class_texts = [f"a photo of a {name}" for name in class_names]
        class_tokens = tokenizer(class_texts)
        class_features = clip.encode_text(class_tokens)
        class_features = F.normalize(class_features, dim=-1)
        # Compute similarities
        similarities = image_features @ class_features.t()
        predictions = similarities.argmax(dim=1)
        print(f"Predictions: {predictions}")
        print(f"Class names: {[class_names[pred] for pred in predictions]}")
