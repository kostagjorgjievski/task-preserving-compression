import torch
import torch.nn as nn


class CNNAutoencoder(nn.Module):
    def __init__(self, input_channels, latent_dim, input_length):
        super().__init__()
        C = input_channels
        L = input_length

        # Encoder: [B, C, L] -> ...
        self.encoder = nn.Sequential(
            nn.Conv1d(C, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
        )

        # compute encoded shape
        with torch.no_grad():
            dummy = torch.zeros(1, C, L)
            enc = self.encoder(dummy)
            self.enc_shape = enc.shape[1:]  # (C_enc, L_enc)
            enc_dim = enc.numel()

        self.to_latent = nn.Linear(enc_dim, latent_dim)
        self.from_latent = nn.Linear(latent_dim, enc_dim)

        # Decoder: mirror using ConvTranspose1d
        C_enc, L_enc = self.enc_shape
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(32, C, kernel_size=4, stride=2, padding=1),
        )

        # If output length is slightly off due to strides, we will crop or pad in forward.

        self.target_length = L

    def encode(self, x):
        # x: [B, L, C] -> [B, C, L]
        x = x.transpose(1, 2)
        h = self.encoder(x)
        h_flat = h.view(h.size(0), -1)
        z = self.to_latent(h_flat)
        return z

    def decode(self, z):
        h_flat = self.from_latent(z)
        B = z.size(0)
        C_enc, L_enc = self.enc_shape
        h = h_flat.view(B, C_enc, L_enc)
        x_rec = self.decoder(h)  # [B, C, L_out]
        x_rec = x_rec.transpose(1, 2)  # [B, L_out, C]

        L_out = x_rec.size(1)
        if L_out > self.target_length:
            x_rec = x_rec[:, : self.target_length, :]
        elif L_out < self.target_length:
            pad_len = self.target_length - L_out
            pad = x_rec[:, -1:, :].repeat(1, pad_len, 1)
            x_rec = torch.cat([x_rec, pad], dim=1)

        return x_rec

    def forward(self, x):
        z = self.encode(x)
        x_rec = self.decode(z)
        return x_rec, z
