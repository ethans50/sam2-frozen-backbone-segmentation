"""Decode Cityscapes gtFine `labelIds` (34-class raw id) into 19-class `trainId`.

id/trainId/category/color all come straight from the `cityscapesscripts` package's
`labels.py` (`pip install cityscapesscripts`) — no hardcoded/remembered values, same
principle as reading class_dict.csv directly for CamVid.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

IGNORE_INDEX = 255


def build_lookup():
    from cityscapesscripts.helpers.labels import labels as cs_labels

    lut = np.full(256, IGNORE_INDEX, dtype=np.uint8)
    trainid_names = [None] * 19
    trainid_colors = [None] * 19
    source_rows = []
    for l in cs_labels:
        if l.id < 0:  # 'license plate' placeholder, never appears as an actual pixel value
            continue
        tid = IGNORE_INDEX if l.trainId == 255 else l.trainId
        lut[l.id] = tid
        source_rows.append({
            "name": l.name, "id": l.id, "trainId": int(tid),
            "category": l.category, "color": list(l.color),
        })
        if tid != IGNORE_INDEX:
            trainid_names[tid] = l.name
            trainid_colors[tid] = list(l.color)

    meta = {
        "classes": trainid_names,
        "colors": trainid_colors,
        "ignore_index": IGNORE_INDEX,
        "source_rows": source_rows,
    }
    return lut, meta


def decode_label(lut: np.ndarray, label_path: Path) -> np.ndarray:
    arr = np.array(Image.open(label_path))  # mode 'L', pixel value = raw id (0-33)
    return lut[arr]


def visualize(img_path: Path, label_path: Path, lut: np.ndarray, meta: dict, out_path: Path):
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    img = np.array(Image.open(img_path).convert("RGB"))
    idx_map = decode_label(lut, label_path)

    colors = meta["colors"] + [[0, 0, 0]]  # ignore -> black
    cmap = np.array(colors, dtype=np.uint8)
    remap = np.zeros(256, dtype=np.uint8)
    for pos in range(19):
        remap[pos] = pos
    remap[IGNORE_INDEX] = 19
    vis = cmap[remap[idx_map]]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), layout="constrained")
    axes[0].imshow(img); axes[0].set_title(img_path.name); axes[0].axis("off")
    axes[1].imshow(vis); axes[1].set_title("decoded label (Cityscapes trainId)"); axes[1].axis("off")
    patches = [mpatches.Patch(color=np.array(c) / 255, label=n) for c, n in zip(meta["colors"], meta["classes"])]
    patches.append(mpatches.Patch(color="black", label="ignore"))
    fig.legend(handles=patches, loc="outside lower center", ncol=7, fontsize=7)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data/cityscapes/raw")
    ap.add_argument("--out-dir", default="data/cityscapes")
    ap.add_argument("--vis-out", default="outputs/figures")
    ap.add_argument("--num-vis", type=int, default=6)
    ap.add_argument("--decode-splits", nargs="+", default=["train", "val"])
    args = ap.parse_args()

    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir)
    vis_out = Path(args.vis_out)
    out_dir.mkdir(parents=True, exist_ok=True)
    vis_out.mkdir(parents=True, exist_ok=True)

    lut, meta = build_lookup()
    np.save(out_dir / "id_lookup.npy", lut)
    with open(out_dir / "class_map.json", "w") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"Cityscapes 19-class (trainId) list: {meta['classes']}")
    print(f"ignore_index: {meta['ignore_index']}")

    # validation-visualization samples (num_vis images from train, mixed cities)
    train_img_dir = data_root / "leftImg8bit" / "train"
    train_lbl_dir = data_root / "gtFine" / "train"
    img_files = sorted(train_img_dir.glob("*/*_leftImg8bit.png"))
    step = max(1, len(img_files) // max(args.num_vis, 1))
    for img_path in img_files[::step][: args.num_vis]:
        city = img_path.parent.name
        stem = img_path.name.replace("_leftImg8bit.png", "")
        label_path = train_lbl_dir / city / f"{stem}_gtFine_labelIds.png"
        if not label_path.exists():
            print(f"warning: label not found, skipping: {label_path}")
            continue
        out_path = vis_out / f"classmap_check_cityscapes_{stem}.png"
        visualize(img_path, label_path, lut, meta, out_path)
        print(f"saved visualization: {out_path}")

    # decode+cache all train/val labels as trainId index PNGs (same pattern as CamVid)
    for split in args.decode_splits:
        img_dir = data_root / "leftImg8bit" / split
        lbl_dir = data_root / "gtFine" / split
        if not img_dir.exists():
            print(f"warning: split not found, skipping: {img_dir}")
            continue
        idx_out_dir = out_dir / "labels_idx" / split
        idx_out_dir.mkdir(parents=True, exist_ok=True)
        n = 0
        for img_path in sorted(img_dir.glob("*/*_leftImg8bit.png")):
            city = img_path.parent.name
            stem = img_path.name.replace("_leftImg8bit.png", "")
            label_path = lbl_dir / city / f"{stem}_gtFine_labelIds.png"
            if not label_path.exists():
                continue
            idx_map = decode_label(lut, label_path)
            Image.fromarray(idx_map, mode="L").save(idx_out_dir / f"{stem}.png")
            n += 1
        print(f"{split}: decoded {n} images to trainId labels -> {idx_out_dir}")


if __name__ == "__main__":
    main()
