from pathlib import Path

import faiss
import numpy as np
import torch
import torch.nn as nn

INPUT_DIMS = 16


class TextureDecoder(nn.Module):
    def __init__(self):
        super(TextureDecoder, self).__init__()
        linear_dims = [INPUT_DIMS, 144, 1296, 2048]
        decoder_layers = []
        for i in range(len(linear_dims) - 1):
            if i > 0:
                decoder_layers.append(nn.ReLU())
            decoder_layers.append(
                nn.Linear(in_features=linear_dims[i], out_features=linear_dims[i + 1])
            )
        decoder_layers.append(nn.Unflatten(1, (8, 16, 16)))
        conv_dims = [8, 5, 3]
        for i in range(len(conv_dims) - 1):
            if i > 0:
                decoder_layers.append(nn.ReLU())
            decoder_layers.append(
                nn.Conv2d(
                    in_channels=conv_dims[i],
                    out_channels=conv_dims[i + 1],
                    kernel_size=3,
                    stride=1,
                    padding=1,
                )
            )
        self.decoder = nn.ModuleList(decoder_layers)

    def load_pretrained(self):
        proj_root = Path(__file__).parent
        checkpoint = torch.load(proj_root / "ckpt/best_ckpt.pth")
        self.load_state_dict(checkpoint)

    def forward(self, x):
        for m in self.decoder:
            x = m(x)
        return x


class EmbeddingLoss(nn.Module):
    def __init__(
        self,
        **kwargs,
    ) -> None:
        super().__init__()
        proj_root = Path(__file__).parent
        self.index = faiss.IndexFlatL2(INPUT_DIMS)
        embeddings = np.load(proj_root / "processed/embeddings.npy")
        self.index.add(embeddings)
        self.embeddings = torch.tensor(embeddings)

    def forward(self, pred_embeddings):
        distances, indices = self.index.search(pred_embeddings.detach().numpy(), 1)
        indices = indices.flatten()
        closest_embeddings = self.embeddings[indices, :]
        return ((pred_embeddings - closest_embeddings) ** 2).mean()
