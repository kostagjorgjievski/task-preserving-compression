import torch
import torch.nn as nn


class ResidualDilatedBlock(nn.Module):
    """
    Residual 1D block with dilation.
    Input and output shapes: [B, C, L].
    """
    def __init__(self, channels: int, dilation: int = 1):
        super().__init__()
        self.conv = nn.Conv1d(
            channels,
            channels,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
        )
        self.activation = nn.ReLU()

    def forward(self, x):
        # x: [B, C, L]
        out = self.conv(x)
        out = self.activation(out)
        return x + out  # simple residual
        

class DilatedCNNAutoencoder(nn.Module):
    """
    Dilated CNN autoencoder for time series.
    Interface matches CNNAutoencoder:
      - __init__(input_channels, latent_dim, input_length)
      - forward(x) where x is [B, L, C], returns (x_rec, z)
    """

    def __init__(self, input_channels: int, latent_dim: int, input_length: int):
        super().__init__()
        C = input_channels
        L = input_length

        # Encoder: [B, C, L] -> [B, C_enc, L_enc]
        # Idea:
        #   1) Start with a regular conv to lift to 64 channels.
        #   2) Use several residual dilated blocks at this resolution.
        #   3) Downsample gradually with stride 2 convs while adding more dilated blocks.
        #
        # This gives a much larger effective receptive field than the plain CNNAutoencoder.

        encoder_layers = [
            nn.Conv1d(C, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),

            # Dilated blocks at full resolution
            ResidualDilatedBlock(64, dilation=1),
            ResidualDilatedBlock(64, dilation=2),

            # Downsample 1: L -> L / 2
            nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),

            # Dilated blocks at half resolution
            ResidualDilatedBlock(128, dilation=2),
            ResidualDilatedBlock(128, dilation=4),

            # Downsample 2: L / 2 -> L / 4
            nn.Conv1d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),

            # Dilated blocks at quarter resolution
            ResidualDilatedBlock(128, dilation=4),
            ResidualDilatedBlock(128, dilation=8),

            # Downsample 3: L / 4 -> L / 8
            nn.Conv1d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),

            # A final dilated block at coarsest scale
            ResidualDilatedBlock(128, dilation=8),
        ]

        self.encoder = nn.Sequential(*encoder_layers)

        # Compute encoded shape using a dummy pass
        with torch.no_grad():
            dummy = torch.zeros(1, C, L)
            enc = self.encoder(dummy)
            self.enc_shape = enc.shape[1:]  # (C_enc, L_enc)
            enc_dim = enc.numel()

        # Latent mapping
        self.to_latent = nn.Linear(enc_dim, latent_dim)
        self.from_latent = nn.Linear(latent_dim, enc_dim)

        # Decoder: mirror with ConvTranspose1d to get back to length L
        C_enc, L_enc = self.enc_shape
        self.decoder = nn.Sequential(
            # Up 1: L_enc -> 2 * L_enc
            nn.ConvTranspose1d(128, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),

            # Some convs at this scale (no dilation needed in decoder)
            nn.Conv1d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),

            # Up 2: -> 4 * L_enc
            nn.ConvTranspose1d(128, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),

            nn.Conv1d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),

            # Up 3: -> 8 * L_enc ~= original length
            nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),

            nn.Conv1d(32, C, kernel_size=3, stride=1, padding=1),
        )

        # Target output length
        self.target_length = L

    def encode(self, x):
        """
        x: [B, L, C]
        Returns:
          z: [B, latent_dim]
        """
        x = x.transpose(1, 2)  # [B, C, L]
        h = self.encoder(x)
        h_flat = h.view(h.size(0), -1)
        z = self.to_latent(h_flat)
        return z

    def decode(self, z):
        """
        z: [B, latent_dim]
        Returns:
          x_rec: [B, L, C]
        """
        h_flat = self.from_latent(z)
        B = z.size(0)
        C_enc, L_enc = self.enc_shape
        h = h_flat.view(B, C_enc, L_enc)

        x_rec = self.decoder(h)  # [B, C, L_out]
        x_rec = x_rec.transpose(1, 2)  # [B, L_out, C]

        # Crop or pad to exactly target_length
        L_out = x_rec.size(1)
        if L_out > self.target_length:
            x_rec = x_rec[:, : self.target_length, :]
        elif L_out < self.target_length:
            pad_len = self.target_length - L_out
            pad = x_rec[:, -1:, :].repeat(1, pad_len, 1)
            x_rec = torch.cat([x_rec, pad], dim=1)

        return x_rec

    def forward(self, x):
        """
        x: [B, L, C]
        Returns:
          x_rec: [B, L, C]
          z: [B, latent_dim]
        """
        z = self.encode(x)
        x_rec = self.decode(z)
        return x_rec, z
