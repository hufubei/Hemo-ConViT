# src/model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from torch.cuda.amp import autocast

class ResidualBlock(nn.Module):
    """A simple residual block for the regression head."""
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.BatchNorm1d(dim)
        self.linear1 = nn.Linear(dim, dim)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(0.1)
        self.linear2 = nn.Linear(dim, dim)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = self.linear1(x)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x + residual

class ViTForRegressionContrastive(nn.Module):
    """Vision Transformer model with regression and contrastive projection heads."""
    def __init__(self, model_name='deit_small_patch16_224', pretrained=True, num_regression_outputs=1, projection_dim=128):
        super().__init__()
        # Main feature extractor
        self.vit_backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        emb_dim = self.vit_backbone.embed_dim

        # The model has two output heads:
        # 1. regression_head: Predicts the Hgb value.
        # 2. projection_head: Creates embeddings for the contrastive loss during training.
        self.regression_head = nn.Sequential(
            nn.Linear(emb_dim, emb_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            ResidualBlock(emb_dim // 2),
            ResidualBlock(emb_dim // 2),
            nn.Linear(emb_dim // 2, num_regression_outputs)
        )
        self.projection_head = nn.Sequential(
            nn.Linear(emb_dim, emb_dim // 2),
            nn.GELU(),
            nn.Linear(emb_dim // 2, projection_dim)
        )

    def forward(self, x_raw, apply_transform=None):
        # Handle single image inference
        if x_raw.ndim == 3:
            x_raw = x_raw.unsqueeze(0)

        with autocast():
            # Apply GPU-based transforms if provided (e.g., Kornia)
            x_processed = apply_transform(x_raw) if apply_transform else x_raw
            
            # Get feature embedding from the backbone
            emb = self.vit_backbone(x_processed)

            # Get regression output
            hgb_prediction = self.regression_head(emb).squeeze(-1)

            # During training, also return features for contrastive loss.
            # During evaluation, return only the prediction.
            if self.training:
                contrastive_features = self.projection_head(emb)
                return hgb_prediction, contrastive_features
            else:
                return hgb_prediction


class AdaptiveContrastiveLoss(nn.Module):
    """Calculates contrastive loss, weighting sample pairs by Hgb value similarity."""
    def __init__(self, temperature=0.1, hgb_similarity_sigma=1.0):
        super().__init__()
        self.temperature = temperature
        self.hgb_similarity_sigma = hgb_similarity_sigma

    def forward(self, features, hgb_labels):
        # Normalize features to lie on the unit hypersphere
        features = F.normalize(features, p=2, dim=1)
        batch_size = features.size(0)

        # Cosine similarity matrix
        feature_sim_matrix = torch.matmul(features, features.T)

        # Calculate Hgb similarity weights based on a Gaussian kernel
        hgb_dist_matrix = torch.abs(hgb_labels.unsqueeze(1) - hgb_labels.unsqueeze(0))
        hgb_sim_weights = torch.exp(-torch.pow(hgb_dist_matrix, 2) / (2 * self.hgb_similarity_sigma**2))

        # Mask out diagonal (self-similarity)
        diag_mask = ~torch.eye(batch_size, dtype=torch.bool, device=features.device)

        logits = (feature_sim_matrix / self.temperature).masked_fill(~diag_mask, -torch.inf)
        
        # Calculate weighted cross-entropy loss
        log_probs = F.log_softmax(logits, dim=1)
        loss = -torch.sum(hgb_sim_weights[diag_mask] * log_probs[diag_mask]) / diag_mask.sum()
        
        return loss