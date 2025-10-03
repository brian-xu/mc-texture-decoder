import torch.nn as nn


class TextureDecoder(nn.Module):
    def __init__(self):
        super(TextureDecoder, self).__init__()
        linear_dims = [64, 576, 4608, 3072]
        decoder_layers = []
        for i in range(len(linear_dims) - 1):
            if i > 0:
                decoder_layers.append(nn.ReLU())
            decoder_layers.append(
                nn.Linear(in_features=linear_dims[i], out_features=linear_dims[i + 1])
            )
        decoder_layers.append(nn.Unflatten(1, (12, 16, 16)))
        conv_dims = [12, 6, 3]
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

    def forward(self, x):
        for m in self.decoder:
            x = m(x)
        return x
