"""CamVid/Cityscapes Dataset/DataLoader. Labels are read as-is from <data_root>/labels_idx/<split>/*.png
(single-channel index, ignore_index=255), pre-decoded by build_class_map.py / build_class_map_cityscapes.py.
"""
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

SAM2_MEAN = [0.485, 0.456, 0.406]
SAM2_STD = [0.229, 0.224, 0.225]


def _load_pair(img_path: Path, label_path: Path, size_wh: tuple, mean: torch.Tensor, std: torch.Tensor):
    if not label_path.exists():
        raise FileNotFoundError(f"decoded label not found: {label_path}")

    img = Image.open(img_path).convert("RGB").resize(size_wh, Image.BILINEAR)
    label = Image.open(label_path).resize(size_wh, Image.NEAREST)

    img_t = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
    img_t = (img_t - mean) / std
    label_t = torch.from_numpy(np.array(label)).long()
    return img_t, label_t


class CamVidDataset(Dataset):
    """CamVid11, resized to a square (resolution×resolution) — slightly distorts the 4:3 original, validated in Phase 1."""

    def __init__(self, split: str, data_root: str = "data/camvid", resolution: int = 512):
        self.data_root = Path(data_root)
        self.size_wh = (resolution, resolution)
        self.img_dir = self.data_root / "raw" / split
        self.label_dir = self.data_root / "labels_idx" / split
        self.img_paths = sorted(self.img_dir.glob("*.png"))
        if not self.img_paths:
            raise FileNotFoundError(f"no images found: {self.img_dir}")

        with open(self.data_root / "class_map.json") as f:
            meta = json.load(f)
        self.classes = meta["classes"]
        self.ignore_index = meta["ignore_index"]
        self.mean = torch.tensor(SAM2_MEAN).view(3, 1, 1)
        self.std = torch.tensor(SAM2_STD).view(3, 1, 1)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, i):
        img_path = self.img_paths[i]
        label_path = self.label_dir / img_path.name
        return _load_pair(img_path, label_path, self.size_wh, self.mean, self.std)


class CityscapesDataset(Dataset):
    """Cityscapes 19-class (trainId). Resized to half size (1024x512), keeping the original 2:1
    aspect ratio (not squashed to square — see DETAILS.md Phase 2 "해상도").
    Images: leftImg8bit/<split>/<city>/*_leftImg8bit.png. Labels: labels_idx/<split>/<stem>.png
    (flat, no per-city subfolder — that's how build_class_map_cityscapes.py stores them)."""

    def __init__(self, split: str, data_root: str = "data/cityscapes", size_wh: tuple = (1024, 512)):
        self.data_root = Path(data_root)
        self.size_wh = size_wh
        self.img_dir = self.data_root / "raw" / "leftImg8bit" / split
        self.label_dir = self.data_root / "labels_idx" / split
        self.img_paths = sorted(self.img_dir.glob("*/*_leftImg8bit.png"))
        if not self.img_paths:
            raise FileNotFoundError(f"no images found: {self.img_dir}")

        with open(self.data_root / "class_map.json") as f:
            meta = json.load(f)
        self.classes = meta["classes"]
        self.ignore_index = meta["ignore_index"]
        self.mean = torch.tensor(SAM2_MEAN).view(3, 1, 1)
        self.std = torch.tensor(SAM2_STD).view(3, 1, 1)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, i):
        img_path = self.img_paths[i]
        stem = img_path.name.replace("_leftImg8bit.png", "")
        label_path = self.label_dir / f"{stem}.png"
        return _load_pair(img_path, label_path, self.size_wh, self.mean, self.std)


if __name__ == "__main__":
    for name, ds in [("CamVid", CamVidDataset("train")), ("Cityscapes", CityscapesDataset("train"))]:
        print(f"[{name}] num samples: {len(ds)}, classes: {ds.classes}, ignore_index: {ds.ignore_index}")
        img, label = ds[0]
        print(f"  image shape: {img.shape}, dtype: {img.dtype}")
        print(f"  label shape: {label.shape}, dtype: {label.dtype}, unique: {label.unique().tolist()}")
