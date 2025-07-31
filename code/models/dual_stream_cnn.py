"""
The DualStreamCNN model architecture.

This model processes a 1D time-series and a 2D spectrogram in parallel
through two separate backbones. The features are then concatenated and
passed through a final fusion head for classification.
"""

import torch
import torch.nn as nn

from .multi_scale_cnn import MultiScaleCNN

class DualStreamCNN(nn.Module):
    def __init__(self, ts_config: dict, spec_stream_model: nn.Module, spec_feature_dim: int, fusion_config: dict, num_classes: int):
        super().__init__()

        # --- Time-Series Stream ---
        # We reuse MultiScaleCNN as it's a powerful feature extractor.
        ts_config['num_classes'] = fusion_config['ts_feature_dim']
        self.ts_stream = MultiScaleCNN(**ts_config)

        # --- Spectrogram Stream ---
        # This is now a pre-built model (e.g., custom CNN or a pre-trained ResNet)
        self.spec_stream = spec_stream_model
        
        # --- Fusion Head ---
        combined_feature_size = fusion_config['ts_feature_dim'] + spec_feature_dim
        self.fusion_head = nn.Sequential(
            nn.Linear(combined_feature_size, fusion_config['hidden_dim']),
            nn.LayerNorm(fusion_config['hidden_dim']),
            nn.ReLU(),
            nn.Dropout(fusion_config.get('dropout', 0.5)),
            nn.Linear(fusion_config['hidden_dim'], num_classes)
        )

    def forward(self, x_ts: torch.Tensor, x_spec: torch.Tensor) -> torch.Tensor:
        # Process each stream independently to get high-level features
        ts_features = self.ts_stream(x_ts)
        spec_features = self.spec_stream(x_spec)

        # Concatenate the features (late fusion)
        fused_features = torch.cat((ts_features, spec_features), dim=1)

        # Classify using the fusion head
        logits = self.fusion_head(fused_features)
        return logits
