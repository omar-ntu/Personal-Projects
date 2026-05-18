from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


class FrameVAE(nn.Module):
    """Frame-wise VAE baseline.

    This model compresses and reconstructs each frame independently. It is a useful
    baseline because it can preserve spatial appearance but has no explicit temporal
    modelling beyond seeing each frame separately.
    """

    def __init__(self, latent_dim: int = 32, image_size: int = 32):
        super().__init__()
        if image_size != 32:
            raise ValueError("This lightweight implementation assumes image_size=32.")
        self.latent_dim = latent_dim
        self.image_size = image_size
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
        )
        self.enc_dim = 32 * 4 * 4
        self.fc_mu = nn.Linear(self.enc_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.enc_dim, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, self.enc_dim)
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (32, 4, 4)),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(8, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def encode_frames(self, frames: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(frames)
        return self.fc_mu(h), self.fc_logvar(h)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode_frames(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.fc_decode(z))

    def forward(self, video: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # video: [B, 1, T, H, W]
        b, c, t, h, w = video.shape
        frames = video.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        mu, logvar = self.encode_frames(frames)
        z = self.reparameterize(mu, logvar)
        recon_frames = self.decode_frames(z)
        recon = recon_frames.reshape(b, t, c, h, w).permute(0, 2, 1, 3, 4)
        return recon, mu, logvar

    @torch.no_grad()
    def encode_video_mu(self, video: torch.Tensor) -> torch.Tensor:
        b, c, t, h, w = video.shape
        frames = video.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        mu, _ = self.encode_frames(frames)
        return mu.reshape(b, t, self.latent_dim)

    @torch.no_grad()
    def decode_video_latents(self, z: torch.Tensor) -> torch.Tensor:
        # z: [B, T, latent_dim]
        b, t, d = z.shape
        frames = self.decode_frames(z.reshape(b * t, d))
        return frames.reshape(b, t, 1, self.image_size, self.image_size).permute(0, 2, 1, 3, 4)


class TemporalVAE(nn.Module):
    """Temporal VAE using 3D convolutions over [time, height, width].

    Unlike FrameVAE, this model compresses the full clip into one latent vector, so
    its encoder must preserve both spatial appearance and motion information.
    """

    def __init__(self, latent_dim: int = 64, frames: int = 16, image_size: int = 32):
        super().__init__()
        if frames != 16 or image_size != 32:
            raise ValueError("This lightweight implementation assumes frames=16 and image_size=32.")
        self.latent_dim = latent_dim
        self.frames = frames
        self.image_size = image_size
        self.encoder = nn.Sequential(
            nn.Conv3d(1, 8, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(8, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(16, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
        )
        self.enc_shape = (32, 2, 4, 4)
        self.enc_dim = 32 * 2 * 4 * 4
        self.fc_mu = nn.Linear(self.enc_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.enc_dim, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, self.enc_dim)
        self.decoder = nn.Sequential(
            nn.Unflatten(1, self.enc_shape),
            nn.ConvTranspose3d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(16, 8, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(8, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def encode(self, video: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(video)
        return self.fc_mu(h), self.fc_logvar(h)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.fc_decode(z))

    def forward(self, video: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(video)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

    @torch.no_grad()
    def encode_video_mu(self, video: torch.Tensor) -> torch.Tensor:
        mu, _ = self.encode(video)
        return mu

    @torch.no_grad()
    def decode_video_latents(self, z: torch.Tensor) -> torch.Tensor:
        return self.decode(z)
