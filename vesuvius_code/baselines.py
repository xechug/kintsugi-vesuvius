from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import DepthSequenceMixer, _group_norm, init_weights


class ResidualBlock3d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = _group_norm(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = _group_norm(out_channels)
        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv3d(in_channels, out_channels, kernel_size=1)
        )
        self.activation = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        residual = self.skip(x)
        x = self.activation(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        return self.activation(x + residual)


class LearnedDepthHead(nn.Module):
    def __init__(self, channels, num_classes=1):
        super().__init__()
        self.logits = nn.Conv3d(channels, num_classes, kernel_size=1)
        self.attention = nn.Conv3d(channels, 1, kernel_size=1)

    def forward(self, features, return_weights=False):
        logits_3d = self.logits(features)
        weights = torch.softmax(self.attention(features), dim=2)
        logits_2d = (logits_3d * weights).sum(dim=2)
        if return_weights:
            return logits_2d, weights
        return logits_2d


def _pad_replicate_3d(
    x: torch.Tensor,
    pad_z: int = 0,
    pad_y: int = 0,
    pad_x: int = 0,
) -> torch.Tensor:
    if pad_z > 0:
        front = x[:, :, :1].expand(-1, -1, pad_z, -1, -1)
        back = x[:, :, -1:].expand(-1, -1, pad_z, -1, -1)
        x = torch.cat((front, x, back), dim=2)
    if pad_y > 0:
        top = x[:, :, :, :1].expand(-1, -1, -1, pad_y, -1)
        bottom = x[:, :, :, -1:].expand(-1, -1, -1, pad_y, -1)
        x = torch.cat((top, x, bottom), dim=3)
    if pad_x > 0:
        left = x[:, :, :, :, :1].expand(-1, -1, -1, -1, pad_x)
        right = x[:, :, :, :, -1:].expand(-1, -1, -1, -1, pad_x)
        x = torch.cat((left, x, right), dim=4)
    return x


class PhysicalStem3D(nn.Module):
    """
    Stem físico diferencial para tomografía de contraste de fase (XPCT):
    1. Z-score local por parche (invarianza fotométrica entre fragmentos).
    2. Derivada primera suave en Z (interfaz aire-papiro, estilo Holoborodko).
    3. Derivada segunda suave en Z (Laplaciano de fase en la costra de tinta).
    4. Residuo planar de alta frecuencia (patrón de craquelado / micro-fisuras).
    """

    def __init__(self, in_channels: int = 1, eps: float = 1e-5, clamp_std: float = 4.0):
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(in_channels) * 4
        self.eps = float(eps)
        self.clamp_std = float(clamp_std)

        dz_1d = torch.tensor([-1.0, -2.0, 0.0, 2.0, 1.0], dtype=torch.float32) / 8.0
        dzz_1d = torch.tensor([1.0, 0.0, -2.0, 0.0, 1.0], dtype=torch.float32) / 4.0
        smooth_3 = torch.tensor([1.0, 2.0, 1.0], dtype=torch.float32) / 4.0
        smooth_5 = torch.tensor([1.0, 4.0, 6.0, 4.0, 1.0], dtype=torch.float32) / 16.0

        kernel_dz = (
            dz_1d[:, None, None] * smooth_3[None, :, None] * smooth_3[None, None, :]
        ).view(1, 1, 5, 3, 3)
        kernel_dzz = (
            dzz_1d[:, None, None] * smooth_3[None, :, None] * smooth_3[None, None, :]
        ).view(1, 1, 5, 3, 3)
        kernel_planar = (smooth_5[:, None] * smooth_5[None, :]).view(1, 1, 1, 5, 5)

        self.register_buffer("kernel_dz", kernel_dz.contiguous(), persistent=False)
        self.register_buffer("kernel_dzz", kernel_dzz.contiguous(), persistent=False)
        self.register_buffer("kernel_planar", kernel_planar.contiguous(), persistent=False)
        self.norm = _group_norm(self.out_channels, max_groups=4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, z, y, width = x.shape
        mean = x.mean(dim=(-3, -2, -1), keepdim=True)
        var = x.var(dim=(-3, -2, -1), unbiased=False, keepdim=True)
        zscore = ((x - mean) / torch.sqrt(var + self.eps)).clamp(
            -self.clamp_std, self.clamp_std
        )

        flat = zscore.reshape(b * c, 1, z, y, width)
        padded_z = _pad_replicate_3d(flat, pad_z=2, pad_y=1, pad_x=1)
        dz = F.conv3d(padded_z, self.kernel_dz.to(dtype=x.dtype, device=x.device))
        dzz = F.conv3d(padded_z, self.kernel_dzz.to(dtype=x.dtype, device=x.device))

        padded_xy = _pad_replicate_3d(flat, pad_z=0, pad_y=2, pad_x=2)
        lowpass_xy = F.conv3d(
            padded_xy, self.kernel_planar.to(dtype=x.dtype, device=x.device)
        )
        crackle = flat - lowpass_xy

        stacked = torch.cat((flat, dz, dzz, crackle), dim=1).reshape(
            b, c * 4, z, y, width
        )
        return self.norm(stacked)


class SoftSheetUnwarper3D(nn.Module):
    """
    Alineación diferenciable de la lámina de papiro en Z (Soft Sheet Unwarping).
    Estima un desplazamiento vertical suave Δz(y, x) y remuestrea las características
    a lo largo de Z para corregir el alabeo antes del mezclador secuencial.
    """

    def __init__(self, channels: int, max_shift: float = 2.0):
        super().__init__()
        self.channels = int(channels)
        self.max_shift = float(max_shift)
        self.offset_head = nn.Conv3d(self.channels, 1, kernel_size=1)
        nn.init.zeros_(self.offset_head.weight)
        nn.init.zeros_(self.offset_head.bias)
        self.offset_head.weight._no_reinit = True
        self.offset_head.bias._no_reinit = True

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b, c, depth, height, width = features.shape
        if depth <= 1:
            zero_shift = features.new_zeros((b, 1, height, width))
            return features, zero_shift

        scores = self.offset_head(features)
        weights = torch.softmax(scores, dim=2)
        z_coords = torch.arange(
            depth, device=features.device, dtype=features.dtype
        ).view(1, 1, depth, 1, 1)
        z_center = 0.5 * float(depth - 1)
        raw_shift = ((z_coords - z_center) * weights).sum(dim=2, keepdim=True)
        smooth_shift = F.avg_pool3d(
            _pad_replicate_3d(raw_shift, pad_z=0, pad_y=1, pad_x=1),
            kernel_size=(1, 3, 3),
            stride=1,
        )
        shift = self.max_shift * torch.tanh(
            smooth_shift / max(self.max_shift, 1e-4)
        )

        sample_z = (z_coords + shift).clamp(0.0, float(depth - 1))
        z0 = sample_z.floor().long().clamp(0, depth - 1)
        z1 = (z0 + 1).clamp(0, depth - 1)
        alpha = (sample_z - z0.to(dtype=features.dtype)).expand_as(features)

        f0 = torch.gather(features, dim=2, index=z0.expand_as(features))
        f1 = torch.gather(features, dim=2, index=z1.expand_as(features))
        warped = (1.0 - alpha) * f0 + alpha * f1
        return warped, shift.squeeze(2)


class ResidualUNet3DTo2D(nn.Module):
    def __init__(
        self,
        in_channels=1,
        num_classes=1,
        base_channels=8,
        physical_stem=False,
        sheet_unwarp=False,
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.num_classes = int(num_classes)
        self.base_channels = int(base_channels)
        self.physical_stem = bool(physical_stem)
        self.sheet_unwarp = bool(sheet_unwarp)

        self.stem = (
            PhysicalStem3D(in_channels=self.in_channels)
            if self.physical_stem
            else None
        )
        encoder_in = self.stem.out_channels if self.stem is not None else self.in_channels
        self.encoder = ResidualBlock3d(encoder_in, base_channels)
        self.unwarper = (
            SoftSheetUnwarper3D(base_channels) if self.sheet_unwarp else None
        )
        self.down = nn.Conv3d(
            base_channels,
            base_channels * 2,
            kernel_size=(3, 2, 2),
            stride=(1, 2, 2),
            padding=(1, 0, 0),
        )
        self.bottleneck = ResidualBlock3d(base_channels * 2, base_channels * 2)
        self.up = nn.ConvTranspose3d(
            base_channels * 2,
            base_channels,
            kernel_size=(1, 2, 2),
            stride=(1, 2, 2),
        )
        self.decoder = ResidualBlock3d(base_channels * 2, base_channels)
        self.head = LearnedDepthHead(base_channels, num_classes=num_classes)
        self.apply(init_weights)

    def encode_bottleneck(self, image):
        features = self.stem(image) if self.stem is not None else image
        skip = self.encoder(features)
        self._last_sheet_shift = None
        if self.unwarper is not None:
            skip, self._last_sheet_shift = self.unwarper(skip)
        bottleneck = self.bottleneck(self.down(skip))
        return skip, bottleneck

    def decode(self, skip, bottleneck, return_aux=False):
        upsampled = self.up(bottleneck)
        if upsampled.shape[-3:] != skip.shape[-3:]:
            upsampled = F.interpolate(
                upsampled,
                size=skip.shape[-3:],
                mode="trilinear",
                align_corners=False,
            )
        decoded = self.decoder(torch.cat((skip, upsampled), dim=1))
        logits, depth_weights = self.head(decoded, return_weights=True)
        if return_aux:
            aux = {"depth_weights": depth_weights}
            if getattr(self, "_last_sheet_shift", None) is not None:
                aux["sheet_shift_z"] = self._last_sheet_shift
            return logits, aux
        return logits

    def forward(self, image, return_aux=False):
        skip, bottleneck = self.encode_bottleneck(image)
        return self.decode(skip, bottleneck, return_aux=return_aux)

    def predict_proba(self, image):
        return torch.sigmoid(self.forward(image))

    def get_config(self):
        config = {
            "name": self.__class__.__name__,
            "in_channels": self.in_channels,
            "num_classes": self.num_classes,
            "base_channels": self.base_channels,
            "backend": "convolutional",
        }
        if self.physical_stem:
            config["physical_stem"] = True
        if self.sheet_unwarp:
            config["sheet_unwarp"] = True
        return config


class AxialSequenceMixer3D(nn.Module):
    axes = ("z", "y", "x")

    def __init__(self, channels, backend="auto", bidirectional=True):
        super().__init__()
        self.mixers = nn.ModuleDict(
            {
                axis: DepthSequenceMixer(
                    channels,
                    backend=backend,
                    bidirectional=bidirectional,
                )
                for axis in self.axes
            }
        )
        backends = {mixer.backend for mixer in self.mixers.values()}
        if len(backends) != 1:
            raise RuntimeError(f"Axial mixers selected inconsistent backends: {backends}")
        self.backend = next(iter(backends))
        self.norm = _group_norm(channels)
        for mixer in self.mixers.values():
            if mixer.backend == "fallback":
                mixer.mixer.apply(init_weights)
        self.norm.apply(init_weights)

    @staticmethod
    def _sequence(x, axis):
        b, c, z, y, width = x.shape
        if axis == "z":
            return x.permute(0, 3, 4, 2, 1).reshape(b * y * width, z, c)
        if axis == "y":
            return x.permute(0, 2, 4, 3, 1).reshape(b * z * width, y, c)
        if axis == "x":
            return x.permute(0, 2, 3, 4, 1).reshape(b * z * y, width, c)
        raise ValueError(f"Unknown axis: {axis}")

    @staticmethod
    def _volume(sequence, shape, axis):
        b, c, z, y, width = shape
        if axis == "z":
            return sequence.reshape(b, y, width, z, c).permute(0, 4, 3, 1, 2).contiguous()
        if axis == "y":
            return sequence.reshape(b, z, width, y, c).permute(0, 4, 1, 3, 2).contiguous()
        if axis == "x":
            return sequence.reshape(b, z, y, width, c).permute(0, 4, 1, 2, 3).contiguous()
        raise ValueError(f"Unknown axis: {axis}")

    def forward(self, x):
        outputs = []
        for axis in self.axes:
            sequence = self._sequence(x, axis)
            mixed = self.mixers[axis](sequence)
            outputs.append(self._volume(mixed, x.shape, axis))
        return self.norm(torch.stack(outputs, dim=0).mean(dim=0))


class KintsugiAxialUNet3DTo2D(ResidualUNet3DTo2D):
    def __init__(
        self,
        in_channels=1,
        num_classes=1,
        base_channels=8,
        backend="auto",
        bidirectional=True,
        physical_stem=False,
        sheet_unwarp=False,
    ):
        super().__init__(
            in_channels=in_channels,
            num_classes=num_classes,
            base_channels=base_channels,
            physical_stem=physical_stem,
            sheet_unwarp=sheet_unwarp,
        )
        self.axial_mixer = AxialSequenceMixer3D(
            channels=base_channels * 2,
            backend=backend,
            bidirectional=bidirectional,
        )
        self.bidirectional = bool(bidirectional)

    def forward(self, image, return_aux=False):
        skip, bottleneck = self.encode_bottleneck(image)
        bottleneck = self.axial_mixer(bottleneck)
        decoded = self.decode(skip, bottleneck, return_aux=return_aux)
        if not return_aux:
            return decoded
        logits, auxiliary = decoded
        auxiliary.update(
            {
                "mixed_axes": self.axial_mixer.axes,
                "backend": self.axial_mixer.backend,
            }
        )
        return logits, auxiliary

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "name": self.__class__.__name__,
                "backend": self.axial_mixer.backend,
                "bidirectional": self.bidirectional,
                "mixed_axes": list(self.axial_mixer.axes),
            }
        )
        return config


class ResidualPhysicalUNet3DTo2D(ResidualUNet3DTo2D):
    def __init__(
        self,
        in_channels=1,
        num_classes=1,
        base_channels=8,
        physical_stem=True,
        sheet_unwarp=True,
    ):
        super().__init__(
            in_channels=in_channels,
            num_classes=num_classes,
            base_channels=base_channels,
            physical_stem=physical_stem,
            sheet_unwarp=sheet_unwarp,
        )


class KintsugiPhysicalUNet3DTo2D(KintsugiAxialUNet3DTo2D):
    def __init__(
        self,
        in_channels=1,
        num_classes=1,
        base_channels=8,
        backend="auto",
        bidirectional=True,
        physical_stem=True,
        sheet_unwarp=True,
    ):
        super().__init__(
            in_channels=in_channels,
            num_classes=num_classes,
            base_channels=base_channels,
            backend=backend,
            bidirectional=bidirectional,
            physical_stem=physical_stem,
            sheet_unwarp=sheet_unwarp,
        )

