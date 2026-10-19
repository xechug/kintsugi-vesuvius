import torch
import torch.nn as nn
import torch.nn.functional as F


class FableLoss(nn.Module):
    """
    Loss para tinta escasa basada en logits.

    BCEWithLogitsLoss es numericamente mas estable que Sigmoid + BCELoss.
    Tversky permite penalizar falsos negativos mas que falsos positivos sin usar
    una asimetria tan extrema que destruya la precision.
    """

    def __init__(
        self,
        bce_weight=0.5,
        tversky_weight=0.5,
        pos_weight=3.0,
        tversky_alpha=0.3,
        tversky_beta=0.7,
        smooth=1e-5,
    ):
        super().__init__()
        self.bce_weight = float(bce_weight)
        self.tversky_weight = float(tversky_weight)
        self.register_buffer("pos_weight", torch.tensor(float(pos_weight)))
        self.tversky_alpha = float(tversky_alpha)
        self.tversky_beta = float(tversky_beta)
        self.smooth = float(smooth)

    def forward(self, logits, target, valid_mask=None):
        if logits.shape != target.shape:
            target = F.interpolate(target, size=logits.shape[-2:], mode="nearest")

        target = target.to(dtype=logits.dtype)
        if valid_mask is None:
            valid = torch.ones_like(target)
        else:
            if valid_mask.shape != logits.shape:
                valid_mask = F.interpolate(
                    valid_mask.to(dtype=logits.dtype),
                    size=logits.shape[-2:],
                    mode="nearest",
                )
            valid = valid_mask.to(device=logits.device, dtype=logits.dtype)
            valid = (valid > 0.5).to(dtype=logits.dtype)

        bce_per_pixel = F.binary_cross_entropy_with_logits(
            logits,
            target,
            pos_weight=self.pos_weight.to(device=logits.device, dtype=logits.dtype),
            reduction="none",
        )
        bce = (bce_per_pixel * valid).sum() / valid.sum().clamp_min(1.0)

        probs = torch.sigmoid(logits)
        probs_flat = probs.reshape(probs.shape[0], -1)
        target_flat = target.reshape(target.shape[0], -1)
        valid_flat = valid.reshape(valid.shape[0], -1)

        tp = (probs_flat * target_flat * valid_flat).sum(dim=1)
        fp = (probs_flat * (1.0 - target_flat) * valid_flat).sum(dim=1)
        fn = ((1.0 - probs_flat) * target_flat * valid_flat).sum(dim=1)

        tversky = (tp + self.smooth) / (
            tp + self.tversky_alpha * fp + self.tversky_beta * fn + self.smooth
        )
        has_positive_target = (target_flat * valid_flat).sum(dim=1) > 0
        if bool(has_positive_target.any()):
            tversky_loss = 1.0 - tversky[has_positive_target].mean()
        else:
            tversky_loss = logits.sum() * 0.0

        return self.bce_weight * bce + self.tversky_weight * tversky_loss


class BCEDiceLoss(FableLoss):
    """
    Compatibilidad con scripts antiguos. Ahora usa logits internamente.
    """

    def __init__(self, smooth=1e-5, pos_weight=3.0):
        super().__init__(
            bce_weight=0.5,
            tversky_weight=0.5,
            pos_weight=pos_weight,
            tversky_alpha=0.5,
            tversky_beta=0.5,
            smooth=smooth,
        )


class AnisotropicFiberSuppressionLoss(nn.Module):
    """
    Physical Option 3 — Anisotropic Fiber-Texture Suppression.

    Computes directional coherence of papyrus fibers from the 2D structure tensor
    J = G_rho * (∇S ∇S^T) on the surface projection of the volume, penalizing
    predicted probability p = σ(logits) from mimicking anisotropic striations
    of plant fibers in background regions (target == 0).
    """

    def __init__(
        self,
        base_loss: nn.Module | None = None,
        fiber_weight: float = 0.15,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.base_loss = base_loss if base_loss is not None else FableLoss()
        self.fiber_weight = float(fiber_weight)
        self.eps = float(eps)

        sobel_x = torch.tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
            dtype=torch.float32,
        ).view(1, 1, 3, 3) / 8.0
        sobel_y = sobel_x.transpose(-1, -2).contiguous()
        smooth_5 = torch.tensor([1.0, 4.0, 6.0, 4.0, 1.0], dtype=torch.float32) / 16.0
        window_5x5 = (smooth_5[:, None] * smooth_5[None, :]).view(1, 1, 5, 5)

        self.register_buffer("sobel_x", sobel_x, persistent=False)
        self.register_buffer("sobel_y", sobel_y, persistent=False)
        self.register_buffer("window_5x5", window_5x5, persistent=False)

    def _compute_fiber_coherence(self, volume: torch.Tensor) -> torch.Tensor:
        if volume.ndim == 5:
            surf = volume.mean(dim=(1, 2)).unsqueeze(1)
        elif volume.ndim == 4:
            surf = volume.mean(dim=1, keepdim=True)
        else:
            raise ValueError(f"Expected 4D or 5D volume tensor, got shape {volume.shape}")

        surf = (surf - surf.mean(dim=(-2, -1), keepdim=True)) / (
            surf.std(dim=(-2, -1), unbiased=False, keepdim=True) + self.eps
        )
        padded = F.pad(surf, (1, 1, 1, 1), mode="replicate")
        gx = F.conv2d(padded, self.sobel_x.to(dtype=surf.dtype, device=surf.device))
        gy = F.conv2d(padded, self.sobel_y.to(dtype=surf.dtype, device=surf.device))

        win = self.window_5x5.to(dtype=surf.dtype, device=surf.device)
        j_xx = F.conv2d(F.pad(gx * gx, (2, 2, 2, 2), mode="replicate"), win)
        j_yy = F.conv2d(F.pad(gy * gy, (2, 2, 2, 2), mode="replicate"), win)
        j_xy = F.conv2d(F.pad(gx * gy, (2, 2, 2, 2), mode="replicate"), win)

        trace = j_xx + j_yy
        diff = torch.sqrt((j_xx - j_yy).pow(2) + 4.0 * j_xy.pow(2) + self.eps)
        lambda_1 = 0.5 * (trace + diff)
        lambda_2 = 0.5 * (trace - diff).clamp_min(0.0)
        coherence = ((lambda_1 - lambda_2) / (lambda_1 + lambda_2 + self.eps)).pow(2)
        return coherence.clamp(0.0, 1.0)

    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
        volume: torch.Tensor | None = None,
    ) -> torch.Tensor:
        sup_loss = self.base_loss(logits, target, valid_mask=valid_mask)
        if volume is None or self.fiber_weight <= 0.0:
            return sup_loss

        if target.shape != logits.shape:
            target = F.interpolate(target.to(dtype=logits.dtype), size=logits.shape[-2:], mode="nearest")
        else:
            target = target.to(dtype=logits.dtype)

        if valid_mask is None:
            valid = torch.ones_like(target)
        else:
            if valid_mask.shape != logits.shape:
                valid_mask = F.interpolate(
                    valid_mask.to(dtype=logits.dtype), size=logits.shape[-2:], mode="nearest"
                )
            valid = (valid_mask.to(device=logits.device, dtype=logits.dtype) > 0.5).to(
                dtype=logits.dtype
            )

        coherence = self._compute_fiber_coherence(volume.to(device=logits.device, dtype=logits.dtype))
        if coherence.shape[-2:] != logits.shape[-2:]:
            coherence = F.interpolate(coherence, size=logits.shape[-2:], mode="bilinear", align_corners=False)

        probs = torch.sigmoid(logits)
        bg_mask = (1.0 - target.clamp(0.0, 1.0)) * valid
        fiber_penalty = (coherence * probs.pow(2) * bg_mask).sum() / bg_mask.sum().clamp_min(1.0)
        return sup_loss + self.fiber_weight * fiber_penalty

