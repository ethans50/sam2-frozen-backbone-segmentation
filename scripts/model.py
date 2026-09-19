"""SAM2Backbone (frozen) + lightweight multi-scale decoder.

Decoder design follows DETAILS.md "Decoder 설계":
1) project each scale to a common channel count (proj_dim) via 1x1 conv
2) bilinear-upsample the rest to the highest-resolution scale
3) concat, then 2-3 conv layers
4) final 1x1 conv to num_classes channels
5) bilinear-upsample to input resolution, then compute loss (loss itself lives in train.py)

Caution: running this file via `python -c` or a REPL with cwd=camvid_seg_ws (=parent of sam2/)
makes `import sam2` collide with the vendored repo directory (see CLAUDE.md "환경 관리").
Always run it as a script, e.g. `python scripts/model.py`.
"""
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# checkpoint/config filenames taken as-is from sam2/checkpoints/download_ckpts.sh (vendored official repo)
SAM2_VARIANTS = {
    "tiny": ("configs/sam2.1/sam2.1_hiera_t.yaml", "sam2.1_hiera_tiny.pt"),
    "small": ("configs/sam2.1/sam2.1_hiera_s.yaml", "sam2.1_hiera_small.pt"),
    "base_plus": ("configs/sam2.1/sam2.1_hiera_b+.yaml", "sam2.1_hiera_base_plus.pt"),
    "large": ("configs/sam2.1/sam2.1_hiera_l.yaml", "sam2.1_hiera_large.pt"),
}
DEFAULT_CFG, _default_ckpt_name = SAM2_VARIANTS["tiny"]
DEFAULT_CKPT = PROJECT_ROOT / "checkpoints" / _default_ckpt_name


class SAM2Backbone(nn.Module):
    """Fully freezes the SAM2 image encoder and exposes only backbone_fpn (multi-scale feature maps)."""

    def __init__(self, config_file: str = DEFAULT_CFG, ckpt_path=DEFAULT_CKPT, device="cuda", variant: str = None):
        super().__init__()
        from sam2.build_sam import build_sam2  # lazy import: see shadowing caution above

        if variant is not None:
            config_file, ckpt_name = SAM2_VARIANTS[variant]
            ckpt_path = PROJECT_ROOT / "checkpoints" / ckpt_name

        sam2_model = build_sam2(config_file, str(ckpt_path), device=device)
        self.trunk = sam2_model.image_encoder
        self.trunk.eval()
        for p in self.trunk.parameters():
            p.requires_grad_(False)

    def train(self, mode: bool = True):
        # encoder is always pinned to eval; blocks SegModel.train() from overriding this
        return super().train(False)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        out = self.trunk(x)
        return out["backbone_fpn"]


class Decoder(nn.Module):
    def __init__(self, in_channels: list[int], num_classes: int, proj_dim: int = 128, hidden_dim: int = 128):
        super().__init__()
        self.proj = nn.ModuleList([nn.Conv2d(c, proj_dim, kernel_size=1) for c in in_channels])
        self.fuse = nn.Sequential(
            nn.Conv2d(proj_dim * len(in_channels), hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Conv2d(hidden_dim, num_classes, kernel_size=1)

    def forward(self, feats: list[torch.Tensor]) -> torch.Tensor:
        target_hw = feats[0].shape[-2:]  # highest-resolution scale
        projected = [proj(f) for proj, f in zip(self.proj, feats)]
        upsampled = [
            F.interpolate(f, size=target_hw, mode="bilinear", align_corners=False) for f in projected
        ]
        fused = self.fuse(torch.cat(upsampled, dim=1))
        return self.classifier(fused)


class SegModel(nn.Module):
    def __init__(self, num_classes: int, backbone: SAM2Backbone, backbone_channels: list[int]):
        super().__init__()
        self.backbone = backbone
        self.decoder = Decoder(backbone_channels, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        logits = self.decoder(feats)
        logits = F.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return logits


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    backbone = SAM2Backbone(device=device)

    dummy = torch.randn(2, 3, 512, 512, device=device)
    with torch.no_grad():
        feats = backbone(dummy)
    channels = [f.shape[1] for f in feats]
    print("backbone_fpn shapes:", [tuple(f.shape) for f in feats])

    model = SegModel(num_classes=11, backbone=backbone, backbone_channels=channels).to(device)
    logits = model(dummy)
    print("output logits shape:", tuple(logits.shape))
    assert logits.shape == (2, 11, 512, 512)
    print("OK: forward shape check passed")
