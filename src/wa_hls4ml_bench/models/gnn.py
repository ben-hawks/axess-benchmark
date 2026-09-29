"""GATv2 GNN reference model.

Vendored from ``FPGA_GNN_GATv2`` in wa_hls4ml_models/GNN/Models.py (commit 4aff94b).
Parameter names match the published checkpoint's ``model_state_dict``, so it loads with
``strict=True``. The forward pass is unchanged apart from dropping the attention-weight
bookkeeping, which does not affect outputs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_add_pool, global_max_pool, global_mean_pool


class FPGA_GNN_GATv2(nn.Module):
    def __init__(self, node_feature_dim, num_targets, hidden_dim=128, num_gnn_layers=3,
                 num_attention_heads=4, mlp_hidden_dim=64, dropout_rate=0.2,
                 edge_dim=None, concat_heads=True, residual_connections=True, **_):
        super().__init__()
        self.num_gnn_layers = num_gnn_layers
        self.dropout_rate = dropout_rate
        self.residual_connections = residual_connections
        gat_out_dim = hidden_dim * num_attention_heads if concat_heads else hidden_dim

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.convs.append(GATv2Conv(node_feature_dim, hidden_dim, heads=num_attention_heads,
                                    dropout=dropout_rate, edge_dim=edge_dim, concat=concat_heads))
        self.norms.append(nn.LayerNorm(gat_out_dim))
        for _ in range(num_gnn_layers - 1):
            self.convs.append(GATv2Conv(gat_out_dim, hidden_dim, heads=num_attention_heads,
                                        dropout=dropout_rate, edge_dim=edge_dim, concat=concat_heads))
            self.norms.append(nn.LayerNorm(gat_out_dim))

        self.residual_projs = nn.ModuleList()
        if residual_connections:
            self.residual_projs.append(
                nn.Linear(node_feature_dim, gat_out_dim) if node_feature_dim != gat_out_dim else nn.Identity()
            )
            for _ in range(num_gnn_layers - 1):
                self.residual_projs.append(nn.Identity())

        self.final_projection = nn.Linear(gat_out_dim, hidden_dim)
        self.pool_weight = nn.Parameter(torch.ones(3) / 3)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim + 4, mlp_hidden_dim),  # +2 strategy one-hot, +2 io_type one-hot
            nn.LayerNorm(mlp_hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim // 2),
            nn.LayerNorm(mlp_hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(mlp_hidden_dim // 2, num_targets),
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        for i in range(self.num_gnn_layers):
            identity = x
            x = self.convs[i](x, edge_index)
            x = self.norms[i](x)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout_rate, training=self.training)
            if self.residual_connections:
                x = x + self.residual_projs[i](identity)

        x = self.final_projection(x)
        w = F.softmax(self.pool_weight, dim=0)
        g = (w[0] * global_add_pool(x, batch)
             + w[1] * global_mean_pool(x, batch)
             + w[2] * global_max_pool(x, batch))
        g = torch.cat([g, data.strategy, data.io_type], dim=1)
        return self.mlp(g)


def load(checkpoint_path, device="cpu"):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = dict(ckpt["model_config"])
    if cfg.pop("model_type", "FPGA_GNN_GATv2") != "FPGA_GNN_GATv2":
        raise ValueError("checkpoint is not an FPGA_GNN_GATv2 model")
    model = FPGA_GNN_GATv2(concat_heads=True, residual_connections=True, **cfg)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    return model.to(device).eval(), ckpt
