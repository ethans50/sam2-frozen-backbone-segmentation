"""Load a trained decoder, compute mIoU + per-class IoU on a held-out split (default test),
and save side-by-side input/GT/prediction visualizations.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))

from dataset import CamVidDataset  # noqa: E402
from model import SAM2Backbone, SegModel  # noqa: E402


def colorize(idx_map: np.ndarray, classes: list[str], rep_color: dict, ignore_index: int) -> np.ndarray:
    colors = [rep_color[c] for c in classes] + [[0, 0, 0]]
    remap = np.zeros(256, dtype=np.uint8)
    for pos, orig in enumerate(list(range(len(classes))) + [ignore_index]):
        remap[orig] = pos
    return np.array(colors, dtype=np.uint8)[remap[idx_map]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data/camvid")
    ap.add_argument("--resolution", type=int, default=512)
    ap.add_argument("--split", default="test")
    ap.add_argument("--ckpt", default="checkpoints/best_decoder.pt")
    ap.add_argument("--batch-size", type=int, default=24)
    ap.add_argument("--out-json", default="outputs/results_phase1.json")
    ap.add_argument("--vis-out", default="outputs/figures")
    ap.add_argument("--num-vis", type=int, default=6)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    with open(Path(args.data_root) / "class_map.json") as f:
        meta = json.load(f)
    classes = meta["classes"]
    ignore_index = meta["ignore_index"]
    rep_color = {}
    for row in meta["source_rows"]:
        if row["group"] is not None and row["group"] not in rep_color:
            rep_color[row["group"]] = row["rgb"]

    ds = CamVidDataset(args.split, data_root=args.data_root, resolution=args.resolution)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    backbone = SAM2Backbone(device=device)
    with torch.no_grad():
        dummy = torch.randn(1, 3, args.resolution, args.resolution, device=device)
        channels = [f.shape[1] for f in backbone(dummy)]
    model = SegModel(num_classes=len(classes), backbone=backbone, backbone_channels=channels).to(device)
    model.decoder.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.decoder.eval()

    from torchmetrics.classification import MulticlassJaccardIndex

    overall_jaccard = MulticlassJaccardIndex(num_classes=len(classes), ignore_index=ignore_index, average="macro").to(device)
    per_class_jaccard = MulticlassJaccardIndex(num_classes=len(classes), ignore_index=ignore_index, average="none").to(device)

    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            preds = model(imgs).argmax(dim=1)
            overall_jaccard.update(preds, labels)
            per_class_jaccard.update(preds, labels)

    miou = overall_jaccard.compute().item()
    per_class = per_class_jaccard.compute().cpu().numpy().tolist()
    per_class_dict = {c: round(v, 4) for c, v in zip(classes, per_class)}

    print(f"[{args.split}] mIoU = {miou:.4f}")
    for c, v in per_class_dict.items():
        print(f"  {c:12s}: {v:.4f}")

    results = {
        "split": args.split,
        "num_samples": len(ds),
        "mIoU": round(miou, 4),
        "per_class_IoU": per_class_dict,
        "ignore_index": ignore_index,
        "resolution": args.resolution,
        "ckpt": args.ckpt,
    }
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"saved results: {args.out_json}")

    # side-by-side input / GT / prediction visualization
    vis_out = Path(args.vis_out)
    vis_out.mkdir(parents=True, exist_ok=True)
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    vis_indices = list(range(min(args.num_vis, len(ds))))
    with torch.no_grad():
        for i in vis_indices:
            img_t, label_t = ds[i]
            img_path = ds.img_paths[i]
            pred = model(img_t.unsqueeze(0).to(device)).argmax(dim=1)[0].cpu().numpy()
            gt = label_t.numpy()

            # undo normalization to redraw the raw input image for display
            mean = ds.mean.numpy().reshape(3, 1, 1)
            std = ds.std.numpy().reshape(3, 1, 1)
            img_disp = ((img_t.numpy() * std + mean) * 255).clip(0, 255).astype(np.uint8).transpose(1, 2, 0)

            gt_vis = colorize(gt, classes, rep_color, ignore_index)
            pred_vis = colorize(pred, classes, rep_color, ignore_index)

            fig, axes = plt.subplots(1, 3, figsize=(13, 5), layout="constrained")
            axes[0].imshow(img_disp); axes[0].set_title(img_path.name); axes[0].axis("off")
            axes[1].imshow(gt_vis); axes[1].set_title("GT"); axes[1].axis("off")
            axes[2].imshow(pred_vis); axes[2].set_title("Prediction"); axes[2].axis("off")
            patches = [mpatches.Patch(color=np.array(rep_color[c]) / 255, label=c) for c in classes]
            patches.append(mpatches.Patch(color="black", label="Void(ignore)"))
            fig.legend(handles=patches, loc="outside lower center", ncol=6, fontsize=8)
            out_path = vis_out / f"eval_{args.split}_{img_path.stem}.png"
            fig.savefig(out_path, dpi=120)
            plt.close(fig)
            print(f"saved visualization: {out_path}")


if __name__ == "__main__":
    main()
