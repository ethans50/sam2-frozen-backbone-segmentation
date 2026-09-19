"""Read official class_dict.csv (32-class CamVid) and build a CamVid11 (11-class + Void=ignore)
color -> index lookup.

32->11 grouping rule source: MathWorks "Semantic Segmentation Using Deep Learning" example's
camvidPixelLabelIDs() function, stated as "original SegNet training methodology"
(https://www.mathworks.com/help/vision/ug/semantic-segmentation-using-deep-learning.html).
Colors themselves are never hardcoded, always read from class_dict.csv — the only hardcoded
part is the grouping rule ("which group each class name belongs to"), verifiable against the
source above.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image

CAMVID11_CLASSES = [
    "Sky", "Building", "Pole", "Road", "Pavement",
    "Tree", "SignSymbol", "Fence", "Car", "Pedestrian", "Bicyclist",
]

# raw 32-class name -> CamVid11 group name. Void is handled separately as ignore.
GROUP_OF = {
    "Sky": "Sky",
    "Building": "Building", "Wall": "Building", "Tunnel": "Building", "Archway": "Building", "Bridge": "Building",
    "Column_Pole": "Pole", "TrafficCone": "Pole",
    "Road": "Road", "LaneMkgsDriv": "Road", "LaneMkgsNonDriv": "Road",
    "Sidewalk": "Pavement", "ParkingBlock": "Pavement", "RoadShoulder": "Pavement",
    "Tree": "Tree", "VegetationMisc": "Tree",
    "SignSymbol": "SignSymbol", "Misc_Text": "SignSymbol", "TrafficLight": "SignSymbol",
    "Fence": "Fence",
    "Car": "Car", "SUVPickupTruck": "Car", "Truck_Bus": "Car", "Train": "Car", "OtherMoving": "Car",
    "Pedestrian": "Pedestrian", "Child": "Pedestrian", "CartLuggagePram": "Pedestrian", "Animal": "Pedestrian",
    "Bicyclist": "Bicyclist", "MotorcycleScooter": "Bicyclist",
    "Void": None,  # ignore
}

IGNORE_INDEX = 255


def load_class_dict(csv_path: Path) -> list[tuple[str, tuple[int, int, int]]]:
    rows = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["name"].strip()
            rgb = (int(row["r"]), int(row["g"]), int(row["b"]))
            rows.append((name, rgb))
    return rows


def build_lookup(class_rows: list[tuple[str, tuple[int, int, int]]]) -> tuple[np.ndarray, dict]:
    unmapped = [name for name, _ in class_rows if name not in GROUP_OF]
    if unmapped:
        raise ValueError(
            f"class_dict.csv has classes GROUP_OF doesn't know: {unmapped}. "
            "Class names may differ in a new CamVid release — update GROUP_OF."
        )

    used_groups = {GROUP_OF[name] for name, _ in class_rows if GROUP_OF[name] is not None}
    missing_groups = set(CAMVID11_CLASSES) - used_groups
    if missing_groups:
        raise ValueError(f"no class_dict.csv row maps to these CamVid11 groups: {missing_groups}")

    group_to_idx = {name: i for i, name in enumerate(CAMVID11_CLASSES)}

    lut = np.full((256, 256, 256), IGNORE_INDEX, dtype=np.uint8)
    meta = {"classes": CAMVID11_CLASSES, "ignore_index": IGNORE_INDEX, "source_rows": []}
    for name, (r, g, b) in class_rows:
        group = GROUP_OF[name]
        idx = group_to_idx[group] if group is not None else IGNORE_INDEX
        lut[r, g, b] = idx
        meta["source_rows"].append({"name": name, "rgb": [r, g, b], "group": group, "index": int(idx)})

    return lut, meta


def decode_label(lut: np.ndarray, label_path: Path, known_colors: set) -> tuple[np.ndarray, set]:
    """Decode via lut, and also report any color not in class_dict.csv (outside known_colors).

    A color outside known_colors silently falls to IGNORE_INDEX in lut too — treated as Void
    with no warning — so this must flag it, otherwise label-decoding bugs go unnoticed.
    """
    arr = np.array(Image.open(label_path).convert("RGB"))
    unique_colors = {tuple(c) for c in np.unique(arr.reshape(-1, 3), axis=0).tolist()}
    unknown = unique_colors - known_colors
    idx_map = lut[arr[..., 0], arr[..., 1], arr[..., 2]]
    return idx_map, unknown


def visualize(
    img_path: Path, label_path: Path, lut: np.ndarray, meta: dict, known_colors: set, out_path: Path
):
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    img = np.array(Image.open(img_path).convert("RGB"))
    idx_map, unknown = decode_label(lut, label_path, known_colors)
    if unknown:
        print(f"warning: {label_path.name} has colors not in class_dict.csv -> {unknown} (may be wrongly treated as Void)")

    # legend/visualization: pick one representative raw color per CamVid11 class from class_dict.csv
    rep_color = {}
    for row in meta["source_rows"]:
        if row["group"] is not None and row["group"] not in rep_color:
            rep_color[row["group"]] = row["rgb"]
    colors = [rep_color[c] for c in meta["classes"]] + [[0, 0, 0]]  # ignore -> black
    cmap = np.array(colors, dtype=np.uint8)
    lookup_idx = list(range(len(meta["classes"]))) + [IGNORE_INDEX]
    remap = np.zeros(256, dtype=np.uint8)
    for pos, orig in enumerate(lookup_idx):
        remap[orig] = pos
    vis = cmap[remap[idx_map]]

    fig, axes = plt.subplots(1, 2, figsize=(9, 5), layout="constrained")
    axes[0].imshow(img)
    axes[0].set_title(img_path.name)
    axes[0].axis("off")
    axes[1].imshow(vis)
    axes[1].set_title("decoded label (CamVid11)")
    axes[1].axis("off")
    patches = [
        mpatches.Patch(color=np.array(c) / 255, label=name)
        for c, name in zip(colors[:-1], meta["classes"])
    ]
    patches.append(mpatches.Patch(color="black", label="Void(ignore)"))
    fig.legend(handles=patches, loc="outside lower center", ncol=6, fontsize=8)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data/camvid/raw")
    ap.add_argument("--class-dict", default=None, help="default: <data-root>/class_dict.csv")
    ap.add_argument("--out-dir", default="data/camvid")
    ap.add_argument("--vis-out", default="outputs/figures")
    ap.add_argument("--num-vis", type=int, default=6)
    ap.add_argument("--decode-splits", nargs="+", default=["train", "val"])
    args = ap.parse_args()

    data_root = Path(args.data_root)
    class_dict_path = Path(args.class_dict) if args.class_dict else data_root / "class_dict.csv"
    out_dir = Path(args.out_dir)
    vis_out = Path(args.vis_out)
    out_dir.mkdir(parents=True, exist_ok=True)
    vis_out.mkdir(parents=True, exist_ok=True)

    class_rows = load_class_dict(class_dict_path)
    print(f"loaded {len(class_rows)} classes from class_dict.csv: {class_dict_path}")

    lut, meta = build_lookup(class_rows)
    known_colors = {tuple(r["rgb"]) for r in meta["source_rows"]}
    np.save(out_dir / "color_lookup.npy", lut)
    with open(out_dir / "class_map.json", "w") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"saved lookup: {out_dir/'color_lookup.npy'}, meta: {out_dir/'class_map.json'}")
    print(f"CamVid11 classes (index order): {meta['classes']}")
    print(f"ignore_index (Void): {meta['ignore_index']}")

    # validation-visualization samples (num_vis images from train)
    train_img_dir = data_root / "train"
    train_lbl_dir = data_root / "train_labels"
    img_files = sorted(train_img_dir.glob("*.png"))[: args.num_vis]
    for img_path in img_files:
        label_path = train_lbl_dir / f"{img_path.stem}_L.png"
        if not label_path.exists():
            print(f"warning: label not found, skipping: {label_path}")
            continue
        out_path = vis_out / f"classmap_check_{img_path.stem}.png"
        visualize(img_path, label_path, lut, meta, known_colors, out_path)
        print(f"saved visualization: {out_path}")

    # decode+cache all train/val labels as index PNGs (ready for dataset.py to consume)
    all_unknown = set()
    for split in args.decode_splits:
        img_dir = data_root / split
        lbl_dir = data_root / f"{split}_labels"
        if not img_dir.exists():
            print(f"warning: split not found, skipping: {img_dir}")
            continue
        idx_out_dir = out_dir / "labels_idx" / split
        idx_out_dir.mkdir(parents=True, exist_ok=True)
        n = 0
        for img_path in sorted(img_dir.glob("*.png")):
            label_path = lbl_dir / f"{img_path.stem}_L.png"
            if not label_path.exists():
                continue
            idx_map, unknown = decode_label(lut, label_path, known_colors)
            all_unknown |= unknown
            Image.fromarray(idx_map, mode="L").save(idx_out_dir / f"{img_path.stem}.png")
            n += 1
        print(f"{split}: decoded {n} images to index labels -> {idx_out_dir}")

    if all_unknown:
        print(f"\n*** warning: found {len(all_unknown)} colors not in class_dict.csv across the whole dataset: {all_unknown} ***")
        print("*** these colors are all currently treated as ignore_index — verify whether they're actual label errors ***")
    else:
        print("\nall label colors confirmed within class_dict.csv (no unknown colors).")


if __name__ == "__main__":
    main()
