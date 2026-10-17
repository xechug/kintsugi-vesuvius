import torch
import torch.nn as nn
import torch.nn.functional as F


def _group_norm(channels, max_groups=8):
    groups = min(max_groups, channels)
    while channels % groups != 0:
        groups -= 1
    return nn.GroupNorm(groups, channels)


def init_weights(m):
    if isinstance(m, (nn.Conv3d, nn.Conv1d)):
        if not getattr(m.weight, "_no_reinit", False):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        if m.bias is not None and not getattr(m.bias, "_no_reinit", False):
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, (nn.GroupNorm, nn.InstanceNorm3d)):
        if m.weight is not None:
            nn.init.constant_(m.weight, 1)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        if not getattr(m.weight, "_no_reinit", False):
            nn.init.xavier_uniform_(m.weight)
        if m.bias is not None and not getattr(m.bias, "_no_reinit", False):
            nn.init.constant_(m.bias, 0)


class ConvGNAct3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
            _group_norm(out_channels),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class ForensicTextureExtractor(nn.Module):
    """
    Rama local 3D: aprende textura fina de fibras, grietas y posibles restos de tinta.
    GroupNorm evita depender de batches grandes y es menos agresiva que InstanceNorm canal a canal.
    """

    def __init__(self, in_channels=1, base_channels=16):
        super().__init__()
        self.net = nn.Sequential(
            ConvGNAct3d(in_channels, base_channels),
            ConvGNAct3d(base_channels, base_channels * 2),
            ConvGNAct3d(base_channels * 2, base_channels * 4),
        )

    @property
    def out_channels(self):
        return self.net[-1].block[0].out_channels

    def forward(self, x):
        return self.net(x)


class KintsugiAttention(nn.Module):
    """
    Puerta estructural aprendida. A diferencia de la version inicial, no solo amplifica:
    si la mascara ve patron de fibra sospechoso pero no tinta, puede atenuar la rama global.
    """

    def __init__(self, in_channels, attention_strength=0.5):
        super().__init__()
        hidden = max(1, in_channels // 2)
        self.attention_strength = float(attention_strength)
        self.crack_detector = nn.Sequential(
            nn.Conv3d(in_channels, hidden, kernel_size=3, padding=1),
            _group_norm(hidden),
            nn.ReLU(inplace=True),
            nn.Conv3d(hidden, 1, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, forensic_features, mamba_features):
        kintsugi_mask = self.crack_detector(forensic_features)
        gate = 1.0 + self.attention_strength * (2.0 * kintsugi_mask - 1.0)
        attended_features = mamba_features * gate
        return attended_features, kintsugi_mask


class FallbackDepthMixer(nn.Module):
    """
    Fallback PyTorch puro: mezcla secuencias a lo largo de Z para cada pixel XY.
    No pretende ser Mamba oficial; mantiene la API para poder sustituirlo por mamba_ssm en CUDA.
    """

    def __init__(self, d_model, d_conv=5):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.in_proj = nn.Linear(d_model, d_model * 2)
        self.depth_conv = nn.Conv1d(
            d_model,
            d_model,
            kernel_size=d_conv,
            padding=d_conv // 2,
            groups=d_model,
        )
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        residual = x
        x_proj, gate = self.in_proj(self.norm(x)).chunk(2, dim=-1)
        x_proj = self.depth_conv(x_proj.transpose(1, 2)).transpose(1, 2)
        x_proj = F.silu(x_proj) * F.silu(gate)
        return residual + self.out_proj(x_proj)


class DepthSequenceMixer(nn.Module):
    """
    Mezcla profundidad Z por cada coordenada XY.
    Si mamba_ssm esta instalado y use_real_mamba=True/'auto', usa el bloque oficial.
    """

    def __init__(
        self,
        d_model,
        backend="auto",
        bidirectional=True,
        use_real_mamba=None,
    ):
        super().__init__()
        if use_real_mamba is not None:
            backend = {True: "mamba", False: "fallback", "auto": "auto"}[use_real_mamba]
        if backend not in {"auto", "mamba", "fallback"}:
            raise ValueError("backend must be 'auto', 'mamba', or 'fallback'")
        self.requested_backend = backend
        self.bidirectional = bool(bidirectional)
        self.backend_error = None

        if backend in {"auto", "mamba"}:
            try:
                from mamba_ssm import Mamba

                self.mixer = Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2)
                self.backend = "mamba"
            except Exception as error:
                if backend == "mamba":
                    raise RuntimeError(f"mamba backend requested but unavailable: {error}") from error
                self.backend_error = f"{type(error).__name__}: {error}"
                self.mixer = FallbackDepthMixer(d_model)
                self.backend = "fallback"
        else:
            self.mixer = FallbackDepthMixer(d_model)
            self.backend = "fallback"

    @property
    def uses_real_mamba(self):
        return self.backend == "mamba"

    def forward(self, x):
        forward = self.mixer(x)
        if not self.bidirectional:
            return forward
        backward = self.mixer(x.flip(1)).flip(1)
        return 0.5 * (forward + backward)


class VMamba3D(nn.Module):
    """
    Rama global: trata cada columna Z como secuencia y conserva la grilla XY.
    """

    def __init__(
        self,
        in_channels,
        d_model=64,
        backend="auto",
        bidirectional=True,
        use_real_mamba=None,
    ):
        super().__init__()
        self.patch_embed = ConvGNAct3d(in_channels, d_model)
        self.depth_mixer = DepthSequenceMixer(
            d_model=d_model,
            backend=backend,
            bidirectional=bidirectional,
            use_real_mamba=use_real_mamba,
        )

    def forward(self, x):
        x = self.patch_embed(x)
        b, c, z, y, x_size = x.shape

        sequence = x.permute(0, 3, 4, 2, 1).contiguous()
        sequence = sequence.view(b * y * x_size, z, c)
        sequence = self.depth_mixer(sequence)
        sequence = sequence.view(b, y, x_size, z, c)

        return sequence.permute(0, 4, 3, 1, 2).contiguous()


class DepthLogitAggregator(nn.Module):
    """
    Convierte logits 3D en mapa 2D. top-k mean reduce el riesgo de que una sola capa Z
    ruidosa active tinta, sin diluir tanto como una media completa.
    """

    def __init__(self, mode="topk_mean", topk=4):
        super().__init__()
        self.mode = mode
        self.topk = int(topk)

    def forward(self, logits_3d):
        if self.mode == "max":
            return logits_3d.max(dim=2).values
        if self.mode == "mean":
            return logits_3d.mean(dim=2)
        if self.mode != "topk_mean":
            raise ValueError(f"Unknown depth aggregation mode: {self.mode}")

        k = max(1, min(self.topk, logits_3d.shape[2]))
        return logits_3d.topk(k=k, dim=2).values.mean(dim=2)


class DualStreamVMambaForense(nn.Module):
    """
    Kintsugi-Mamba v2.
    Devuelve logits 2D. Usa predict_proba() para mapas de probabilidad calibrables.
    """

    def __init__(
        self,
        in_channels=1,
        num_classes=1,
        base_channels=16,
        depth_topk=4,
        depth_aggregation="topk_mean",
        attention_strength=0.5,
        use_real_mamba="auto",
        mamba_backend=None,
        bidirectional=True,
    ):
        super().__init__()
        if mamba_backend is None:
            mamba_backend = {True: "mamba", False: "fallback", "auto": "auto"}[
                use_real_mamba
            ]
        self.in_channels = int(in_channels)
        self.num_classes = int(num_classes)
        self.base_channels = int(base_channels)
        self.depth_topk = int(depth_topk)
        self.depth_aggregation = str(depth_aggregation)
        self.attention_strength = float(attention_strength)
        self.bidirectional = bool(bidirectional)
        feature_channels = base_channels * 4

        self.forensic_stream = ForensicTextureExtractor(in_channels, base_channels=base_channels)
        self.mamba_stream = VMamba3D(
            in_channels,
            d_model=feature_channels,
            backend=mamba_backend,
            bidirectional=bidirectional,
        )
        self.kintsugi_attention = KintsugiAttention(
            in_channels=feature_channels,
            attention_strength=attention_strength,
        )

        self.fusion = nn.Sequential(
            ConvGNAct3d(feature_channels * 2, feature_channels),
            ConvGNAct3d(feature_channels, feature_channels),
        )
        self.segmentation_head = nn.Sequential(
            ConvGNAct3d(feature_channels, max(base_channels * 2, num_classes)),
            nn.Conv3d(max(base_channels * 2, num_classes), num_classes, kernel_size=1),
        )
        self.depth_aggregator = DepthLogitAggregator(mode=depth_aggregation, topk=depth_topk)

        self.forensic_stream.apply(init_weights)
        self.mamba_stream.patch_embed.apply(init_weights)
        if self.mamba_stream.depth_mixer.backend == "fallback":
            self.mamba_stream.depth_mixer.mixer.apply(init_weights)
        self.kintsugi_attention.apply(init_weights)
        self.fusion.apply(init_weights)
        self.segmentation_head.apply(init_weights)

    def get_config(self):
        return {
            "name": self.__class__.__name__,
            "in_channels": self.in_channels,
            "num_classes": self.num_classes,
            "base_channels": self.base_channels,
            "depth_topk": self.depth_topk,
            "depth_aggregation": self.depth_aggregation,
            "attention_strength": self.attention_strength,
            "backend": self.mamba_stream.depth_mixer.backend,
            "requested_backend": self.mamba_stream.depth_mixer.requested_backend,
            "bidirectional": self.bidirectional,
        }

    def forward(self, x, return_aux=False):
        f_feat = self.forensic_stream(x)
        m_feat = self.mamba_stream(x)
        m_feat_attended, kintsugi_mask = self.kintsugi_attention(f_feat, m_feat)

        fused = torch.cat([f_feat, m_feat_attended], dim=1)
        fused = self.fusion(fused) + m_feat
        logits_3d = self.segmentation_head(fused)
        logits_2d = self.depth_aggregator(logits_3d)

        if return_aux:
            return logits_2d, {
                "logits_3d": logits_3d,
                "kintsugi_mask": kintsugi_mask,
                "uses_real_mamba": self.mamba_stream.depth_mixer.uses_real_mamba,
                "backend": self.mamba_stream.depth_mixer.backend,
            }
        return logits_2d

    def predict_proba(self, x):
        return torch.sigmoid(self.forward(x))


if __name__ == "__main__":
    model = DualStreamVMambaForense(in_channels=1, use_real_mamba=False)
    dummy_input = torch.randn(1, 1, 32, 64, 64)
    logits, aux = model(dummy_input, return_aux=True)
    print(f"input: {tuple(dummy_input.shape)}")
    print(f"logits: {tuple(logits.shape)}")
    print(f"real_mamba: {aux['uses_real_mamba']}")
