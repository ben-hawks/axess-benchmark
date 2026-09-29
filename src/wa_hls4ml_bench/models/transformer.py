"""Transformer reference model.

Vendored from wa_hls4ml_models/transformer/model.py (commit 4aff94b), unchanged.
The published ``transformer_best_model.pt`` is a bare state_dict for the defaults below.
"""

import torch
import torch.nn as nn


class LayerTokenizer(nn.Module):
    def __init__(self, feature_dim: int = 33, embed_dim: int = 512, max_layers: int = 128):
        super().__init__()
        self.linear = nn.Linear(feature_dim, embed_dim)
        self.pos_emb = nn.Parameter(torch.randn(max_layers, embed_dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))

    def forward(self, x):
        B, L, _ = x.shape
        tokens = self.linear(x) + self.pos_emb[:L, :].unsqueeze(0)
        cls = self.cls_token.expand(B, -1, -1)
        return torch.cat([cls, tokens], dim=1)


class TransformerRegressor(nn.Module):
    def __init__(self, feature_dim=33, embed_dim=512, num_heads=8, ff_dim=512, num_layers=2,
                 output_dim=6, max_layers=51, dropout=0.1):
        super().__init__()
        self.max_layers = max_layers
        self.tokenizer = LayerTokenizer(feature_dim, embed_dim, max_layers)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=ff_dim,
            dropout=dropout, batch_first=False,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.head = nn.Linear(embed_dim, output_dim)

    def forward(self, x, pad_mask):
        """x: (B, L, feature_dim); pad_mask: (B, L), True = padding."""
        B, L, _ = x.shape
        tokens = self.tokenizer(x)
        cls_pad = torch.zeros(B, 1, dtype=torch.bool, device=x.device)
        mask = torch.cat([cls_pad, pad_mask], dim=1)
        out = self.transformer(tokens.transpose(0, 1), src_key_padding_mask=mask)
        return self.head(out[0])


def load(checkpoint_path, device="cpu"):
    model = TransformerRegressor()
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()
