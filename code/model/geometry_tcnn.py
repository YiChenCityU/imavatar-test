import numpy as np
import tinycudann as tcnn
import torch
import torch.nn as nn

class GeometryNetwork(nn.Module):
    def __init__(
            self,
            feature_vector_size,
            d_in,
            condition_in,
            d_out,
            dims,
            hidden_dim=64,
            hash=True,
            n_levels=16,
            log2_hashmap_size=19,
            base_resolution=16,
            smoothstep=False,
    ):
        super().__init__()

        dims = [d_in + condition_in] + dims + [d_out + feature_vector_size]

        self.encoder = tcnn.Encoding(3, {
            "otype": "HashGrid" if hash else "DenseGrid",
            "n_levels": n_levels,
            "n_features_per_level": 2,
            "log2_hashmap_size": log2_hashmap_size,
            "base_resolution": base_resolution,
            "per_level_scale": 1.5,
            "interpolation": "Smoothstep" if smoothstep else "Linear"
        })
        dims[0] = d_in + self.encoder.n_output_dims

        self.condition_in = condition_in

        self.num_layers = len(dims)-2

        # self.geo_net = tcnn.Network(
        #     n_input_dims=dims[0] + condition_in,
        #     n_output_dims=1 + feature_vector_size,
        #     network_config={
        #         "otype": "FullyFusedMLP",
        #         "activation": "ReLU",
        #         "output_activation": "None",
        #         "n_neurons": hidden_dim,
        #         "n_hidden_layers": self.num_layers,
        #     },
        # )
        self.geo_net = nn.Sequential(
            # Input layer
            nn.Linear(dims[0] + condition_in, hidden_dim),
            nn.ReLU(),
            # Hidden layers
            *[
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU()
                ) for _ in range(self.num_layers - 1)
            ],
            # Output layer
            nn.Linear(hidden_dim, 1 + feature_vector_size)
        )

    def forward(self, pnts, condition):
        xyz = pnts
        # min_val = torch.min(xyz)
        # max_val = torch.max(xyz)
        # xyz = (xyz - min_val) / (max_val - min_val)
        pnts = self.encoder(pnts)
        if self.condition_in > 0:
            # Currently only support batch_size=1
            # This is because the current implementation of masking in ray tracing doesn't support other batch sizes.
            num_pixels = int(pnts.shape[0] / condition.shape[0])
            condition = condition.unsqueeze(1).expand(-1, num_pixels, -1).reshape(-1, self.condition_in)
            pnts = torch.cat([pnts, condition], dim=1)
        x = torch.cat([pnts, xyz], dim=1)
        h = self.geo_net(x)
        return h

    def gradient(self, x, condition):
        with torch.enable_grad():
            x.requires_grad_(True)
            y = self.forward(x, condition)[:, :1]
            d_output = torch.ones_like(y, requires_grad=False, device=y.device)
            gradients = torch.autograd.grad(
                outputs=y,
                inputs=x,
                grad_outputs=d_output,
                create_graph=True,
                retain_graph=True,
                only_inputs=True)[0]
            print('----gridents shape: ', gradients.shape)
        return gradients

