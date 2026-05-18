from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F


def vae_loss(recon: torch.Tensor, target: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor, beta: float = 1e-3) -> Dict[str, torch.Tensor]:
    recon_loss = F.mse_loss(recon, target, reduction="mean")
    # Average KL across samples. For FrameVAE, each frame is treated as a sample.
    kl = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
    total = recon_loss + beta * kl
    return {"loss": total, "recon": recon_loss, "kl": kl}


def psnr_from_mse(mse: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return -10.0 * torch.log10(torch.clamp(mse, min=eps))


def simple_ssim(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Global SSIM approximation for grayscale videos in [0, 1].

    This is intentionally dependency-free. It is not a drop-in replacement for
    windowed SSIM used in image-quality papers, but it is useful for comparing runs.
    """
    dims = tuple(range(1, x.ndim))
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    mux = x.mean(dim=dims)
    muy = y.mean(dim=dims)
    varx = ((x - mux.view(-1, *([1] * (x.ndim - 1)))) ** 2).mean(dim=dims)
    vary = ((y - muy.view(-1, *([1] * (y.ndim - 1)))) ** 2).mean(dim=dims)
    cov = ((x - mux.view(-1, *([1] * (x.ndim - 1)))) * (y - muy.view(-1, *([1] * (y.ndim - 1))))).mean(dim=dims)
    ssim = ((2 * mux * muy + c1) * (2 * cov + c2)) / ((mux.pow(2) + muy.pow(2) + c1) * (varx + vary + c2))
    return ssim.mean()


def temporal_difference_mse(recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """MSE between consecutive-frame differences.

    Lower means the reconstructed clip better preserves motion/change over time.
    """
    recon_dt = recon[:, :, 1:] - recon[:, :, :-1]
    target_dt = target[:, :, 1:] - target[:, :, :-1]
    return F.mse_loss(recon_dt, target_dt, reduction="mean")


def reconstruction_metrics(recon: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    mse = F.mse_loss(recon, target, reduction="mean")
    psnr = psnr_from_mse(mse)
    return {
        "mse": float(mse.detach().cpu()),
        "psnr": float(psnr.detach().cpu()),
        "ssim_global": float(simple_ssim(recon.detach(), target.detach()).cpu()),
        "temporal_mse": float(temporal_difference_mse(recon.detach(), target.detach()).cpu()),
    }
