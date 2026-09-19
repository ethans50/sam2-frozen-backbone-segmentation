"""Separate from the 6-image representative sample eval.py/eval_phase2.py save, this scores the
full held-out set per image and visualizes the highest/lowest mIoU cases. A qualitative check to
show both good and bad cases side by side; its scoring differs from the official metric in
results*.json (pixel-wide macro mIoU) — here only classes actually present in each image are
averaged — so it's a ranking aid only, not an official number.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))

from dataset import CamVidDataset  # noqa: E402
from model import SAM2Backbone, SegModel  # noqa: E402


def per_image_miou(pred: np.ndarray, gt: np.ndarray, num_classes: int, ignore_index: int) -> float:
    valid = gt != ignore_index
    ious = []
    for c in range(num_classes):
        gt_c = (gt == c) & valid
        if not gt_c.any():
            continue
        pred_c = (pred == c) & valid
        inter = (gt_c & pred_c).sum()
        union = (gt_c | pred_c).sum()
        ious.append(inter / union if union > 0 else 0.0)
    return float(np.mean(ious)) if ious else float("nan")


def render(img_disp, gt_vis, pred_vis, title, patches, out_path, iou):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(13, 5), layout="constrained")
    axes[0].imshow(img_disp); axes[0].set_title(f"{title}\n(image mIoU={iou:.3f})"); axes[0].axis("off")
    axes[1].imshow(gt_vis); axes[1].set_title("GT"); axes[1].axis("off")
    axes[2].imshow(pred_vis); axes[2].set_title("Prediction"); axes[2].axis("off")
    fig.legend(handles=patches, loc="outside lower center", ncol=6, fontsize=8)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"saved visualization: {out_path}")


def colorize(idx_map: np.ndarray, num_classes: int, colors: list, ignore_index: int) -> np.ndarray:
    remap = np.zeros(256, dtype=np.uint8)
    for pos, orig in enumerate(list(range(num_classes)) + [ignore_index]):
        remap[orig] = pos
    return np.array(colors, dtype=np.uint8)[remap[idx_map]]


def run_phase1(args, device, top_k):
    with open(Path(args.camvid_root) / "class_map.json") as f:
        meta = json.load(f)
    classes = meta["classes"]
    ignore_index = meta["ignore_index"]
    rep_color = {}
    for row in meta["source_rows"]:
        if row["group"] is not None and row["group"] not in rep_color:
            rep_color[row["group"]] = row["rgb"]
    colors = [rep_color[c] for c in classes] + [[0, 0, 0]]

    ds = CamVidDataset("test", data_root=args.camvid_root, resolution=args.camvid_resolution)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    backbone = SAM2Backbone(device=device)
    with torch.no_grad():
        dummy = torch.randn(1, 3, args.camvid_resolution, args.camvid_resolution, device=device)
        channels = [f.shape[1] for f in backbone(dummy)]
    model = SegModel(num_classes=len(classes), backbone=backbone, backbone_channels=channels).to(device)
    model.decoder.load_state_dict(torch.load(args.phase1_ckpt, map_location=device))
    model.decoder.eval()

    scores = []
    idx = 0
    with torch.no_grad():
        for imgs, labels in loader:
            preds = model(imgs.to(device)).argmax(dim=1).cpu().numpy()
            labels_np = labels.numpy()
            for b in range(preds.shape[0]):
                iou = per_image_miou(preds[b], labels_np[b], len(classes), ignore_index)
                scores.append((idx, iou))
                idx += 1

    scores.sort(key=lambda t: t[1])
    worst = scores[:top_k]
    best = scores[-top_k:][::-1]

    vis_out = Path(args.vis_out)
    vis_out.mkdir(parents=True, exist_ok=True)
    patches = _patches(classes, rep_color)

    with torch.no_grad():
        for tag, picks in (("best", best), ("worst", worst)):
            for rank, (i, iou) in enumerate(picks, start=1):
                img_t, label_t = ds[i]
                img_path = ds.img_paths[i]
                pred = model(img_t.unsqueeze(0).to(device)).argmax(dim=1)[0].cpu().numpy()
                gt = label_t.numpy()
                img_disp = _denorm(img_t, ds)
                gt_vis = colorize(gt, len(classes), colors, ignore_index)
                pred_vis = colorize(pred, len(classes), colors, ignore_index)
                out_path = vis_out / f"eval_test_{tag}{rank}_{img_path.stem}.png"
                render(img_disp, gt_vis, pred_vis, img_path.name, patches, out_path, iou)

    print(f"\n[Phase1 CamVid test, {len(ds)} images] per-image mIoU range: {scores[0][1]:.4f} ~ {scores[-1][1]:.4f}")


def run_phase2(args, device, top_k):
    with open(Path(args.cityscapes_root) / "class_map.json") as f:
        num_classes = len(json.load(f)["classes"])
    with open(args.category_map) as f:
        cat_meta = json.load(f)
    categories = cat_meta["categories"]
    ignore_index = cat_meta["ignore_index"]
    cs_t2c = np.array(cat_meta["cityscapes_trainid_to_category"])
    cv_c2c = np.array(cat_meta["camvid_class_to_category"])
    cv_c2c_vis = np.where(cv_c2c == ignore_index, len(categories), cv_c2c)  # same visualization workaround as eval_phase2.py

    from cityscapesscripts.helpers.labels import labels as cs_labels
    cat_color = {}
    for l in cs_labels:
        if l.category in categories and l.category not in cat_color:
            cat_color[l.category] = list(l.color)
    colors = [cat_color[c] for c in categories] + [[0, 0, 0]]

    backbone = SAM2Backbone(device=device, variant="large")
    with torch.no_grad():
        dummy = torch.randn(1, 3, args.height, args.width, device=device)
        channels = [f.shape[1] for f in backbone(dummy)]
    model = SegModel(num_classes=num_classes, backbone=backbone, backbone_channels=channels).to(device)
    model.decoder.load_state_dict(torch.load(args.phase2_ckpt, map_location=device))
    model.decoder.eval()

    splits = [CamVidDataset(s, data_root=args.camvid_root, resolution=args.camvid_resolution) for s in ("train", "val", "test")]
    ds = ConcatDataset(splits)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    scores = []
    idx = 0
    with torch.no_grad():
        for imgs, labels in loader:
            preds_trainid = model(imgs.to(device)).argmax(dim=1).cpu().numpy()
            labels_np = labels.numpy()
            for b in range(preds_trainid.shape[0]):
                pred_cat = cs_t2c[preds_trainid[b]]
                valid_gt = labels_np[b] != ignore_index
                gt_cat = np.full_like(labels_np[b], ignore_index)
                gt_cat[valid_gt] = cv_c2c[labels_np[b][valid_gt]]
                iou = per_image_miou(pred_cat, gt_cat, len(categories), ignore_index)
                scores.append((idx, iou))
                idx += 1

    scores.sort(key=lambda t: t[1])
    worst = scores[:top_k]
    best = scores[-top_k:][::-1]

    def _img_path_at(global_i):
        off = 0
        for s in splits:
            if global_i < off + len(s):
                return s, global_i - off
            off += len(s)
        raise IndexError(global_i)

    import matplotlib.patches as mpatches
    patches = [mpatches.Patch(color=np.array(c) / 255, label=n) for c, n in zip(colors[:-1], categories)]
    patches.append(mpatches.Patch(color="black", label="ignore"))

    vis_out = Path(args.vis_out)
    vis_out.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for tag, picks in (("best", best), ("worst", worst)):
            for rank, (i, iou) in enumerate(picks, start=1):
                split_ds, local_i = _img_path_at(i)
                img_t, label_t = split_ds[local_i]
                img_path = split_ds.img_paths[local_i]
                pred_trainid = model(img_t.unsqueeze(0).to(device)).argmax(dim=1)[0].cpu().numpy()
                gt = label_t.numpy()
                pred_cat = cs_t2c[pred_trainid]
                gt_cat = np.where(gt == ignore_index, len(categories), cv_c2c_vis[np.clip(gt, 0, len(cv_c2c) - 1)])
                img_disp = _denorm(img_t, split_ds)
                out_path = vis_out / f"cross_eval_{tag}{rank}_{img_path.stem}.png"
                render(img_disp, colors_lookup(colors)[gt_cat], colors_lookup(colors)[pred_cat],
                       img_path.name, patches, out_path, iou)

    print(f"\n[Phase2 CamVid cross-eval, {len(ds)} images, 7-category] per-image mIoU range: {scores[0][1]:.4f} ~ {scores[-1][1]:.4f}")


def colors_lookup(colors):
    return np.array(colors, dtype=np.uint8)


def _denorm(img_t, ds):
    mean = ds.mean.numpy().reshape(3, 1, 1)
    std = ds.std.numpy().reshape(3, 1, 1)
    return ((img_t.numpy() * std + mean) * 255).clip(0, 255).astype(np.uint8).transpose(1, 2, 0)


def _patches(classes, rep_color):
    import matplotlib.patches as mpatches
    patches = [mpatches.Patch(color=np.array(rep_color[c]) / 255, label=c) for c in classes]
    patches.append(mpatches.Patch(color="black", label="Void(ignore)"))
    return patches


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["1", "2", "both"], default="both")
    ap.add_argument("--camvid-root", default="data/camvid")
    ap.add_argument("--camvid-resolution", type=int, default=512)
    ap.add_argument("--cityscapes-root", default="data/cityscapes")
    ap.add_argument("--category-map", default="data/category_map.json")
    ap.add_argument("--phase1-ckpt", default="checkpoints/best_decoder.pt")
    ap.add_argument("--phase2-ckpt", default="checkpoints/best_decoder_cityscapes.pt")
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--height", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--vis-out", default="outputs/figures")
    ap.add_argument("--top-k", type=int, default=5)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.phase in ("1", "both"):
        run_phase1(args, device, args.top_k)
        torch.cuda.empty_cache()  # reclaim VRAM before switching tiny -> large model (8GB limit, see CLAUDE.md)
    if args.phase in ("2", "both"):
        run_phase2(args, device, args.top_k)


if __name__ == "__main__":
    main()
