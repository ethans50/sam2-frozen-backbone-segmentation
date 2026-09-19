"""Phase 2 final eval: (1) 19-class mIoU on Cityscapes val (secondary, for standard-benchmark
comparison), (2) 7-category cross-eval mIoU on CamVid (701 images, fully held out from this
decoder's training) — see DETAILS.md Phase 2 "학습/평가 전략", this is the Phase's key result.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import ConcatDataset, DataLoader

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))

from dataset import CamVidDataset, CityscapesDataset  # noqa: E402
from model import SAM2Backbone, SegModel  # noqa: E402


def load_model(ckpt_path: str, num_classes: int, height: int, width: int, device: str) -> SegModel:
    backbone = SAM2Backbone(device=device, variant="large")
    with torch.no_grad():
        dummy = torch.randn(1, 3, height, width, device=device)
        channels = [f.shape[1] for f in backbone(dummy)]
    model = SegModel(num_classes=num_classes, backbone=backbone, backbone_channels=channels).to(device)
    model.decoder.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.decoder.eval()
    return model


def eval_cityscapes_native(model, args, device):
    from torchmetrics.classification import MulticlassJaccardIndex

    ds = CityscapesDataset("val", data_root=args.cityscapes_root, size_wh=(args.width, args.height))
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=4)
    overall = MulticlassJaccardIndex(num_classes=len(ds.classes), ignore_index=ds.ignore_index, average="macro").to(device)
    per_class = MulticlassJaccardIndex(num_classes=len(ds.classes), ignore_index=ds.ignore_index, average="none").to(device)

    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            preds = model(imgs).argmax(dim=1)
            overall.update(preds, labels)
            per_class.update(preds, labels)

    miou = overall.compute().item()
    per_class_vals = per_class.compute().cpu().numpy().tolist()
    return miou, dict(zip(ds.classes, [round(v, 4) for v in per_class_vals])), len(ds)


def eval_camvid_cross(model, args, device, cat_meta):
    from torchmetrics.classification import MulticlassJaccardIndex

    # CamVid was never used to train this model, so use all 3 splits (701 images) for cross-eval.
    ds = ConcatDataset([CamVidDataset(s, data_root=args.camvid_root, resolution=args.camvid_resolution) for s in ("train", "val", "test")])
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    categories = cat_meta["categories"]
    ignore_index = cat_meta["ignore_index"]
    cs_t2c = torch.tensor(cat_meta["cityscapes_trainid_to_category"], device=device)
    cv_c2c = torch.tensor(cat_meta["camvid_class_to_category"], device=device)

    overall = MulticlassJaccardIndex(num_classes=len(categories), ignore_index=ignore_index, average="macro").to(device)
    per_cat = MulticlassJaccardIndex(num_classes=len(categories), ignore_index=ignore_index, average="none").to(device)

    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            preds_trainid = model(imgs).argmax(dim=1)
            preds_cat = cs_t2c[preds_trainid]  # argmax is always 0-18, no ignore handling needed

            valid_gt = labels != ignore_index  # GT may be 255 (ignore), must filter before lookup
            labels_cat = torch.full_like(labels, ignore_index)
            labels_cat[valid_gt] = cv_c2c[labels[valid_gt]]

            overall.update(preds_cat, labels_cat)
            per_cat.update(preds_cat, labels_cat)

    miou = overall.compute().item()
    per_cat_vals = per_cat.compute().cpu().numpy().tolist()
    return miou, dict(zip(categories, [round(v, 4) for v in per_cat_vals])), len(ds)


def visualize_cross_samples(model, args, device, cat_meta, num_vis=6):
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from cityscapesscripts.helpers.labels import labels as cs_labels

    categories = cat_meta["categories"]
    ignore_index = cat_meta["ignore_index"]
    cs_t2c = np.array(cat_meta["cityscapes_trainid_to_category"])
    cv_c2c = np.array(cat_meta["camvid_class_to_category"])
    # visualization only: draw items whose category is ignore(255) (e.g. Bicyclist) in the "ignore" color (len(categories))
    cv_c2c_vis = np.where(cv_c2c == ignore_index, len(categories), cv_c2c)

    cat_color = {}
    for l in cs_labels:
        if l.category in categories and l.category not in cat_color:
            cat_color[l.category] = list(l.color)
    colors = [cat_color[c] for c in categories] + [[0, 0, 0]]
    cmap = np.array(colors, dtype=np.uint8)

    ds = CamVidDataset("test", data_root=args.camvid_root, resolution=args.camvid_resolution)
    vis_out = Path(args.vis_out)
    vis_out.mkdir(parents=True, exist_ok=True)

    step = max(1, len(ds) // num_vis)
    with torch.no_grad():
        for i in range(0, len(ds), step):
            img_t, label_t = ds[i]
            img_path = ds.img_paths[i]
            pred_trainid = model(img_t.unsqueeze(0).to(device)).argmax(dim=1)[0].cpu().numpy()
            gt = label_t.numpy()

            pred_cat = cs_t2c[pred_trainid]  # argmax is always 0-18
            gt_cat = np.where(gt == ignore_index, len(categories), cv_c2c_vis[np.clip(gt, 0, 10)])

            mean = ds.mean.numpy().reshape(3, 1, 1)
            std = ds.std.numpy().reshape(3, 1, 1)
            img_disp = ((img_t.numpy() * std + mean) * 255).clip(0, 255).astype(np.uint8).transpose(1, 2, 0)

            fig, axes = plt.subplots(1, 3, figsize=(13, 5), layout="constrained")
            axes[0].imshow(img_disp); axes[0].set_title(img_path.name); axes[0].axis("off")
            axes[1].imshow(cmap[gt_cat]); axes[1].set_title("GT (category)"); axes[1].axis("off")
            axes[2].imshow(cmap[pred_cat]); axes[2].set_title("Prediction (category)"); axes[2].axis("off")
            patches = [mpatches.Patch(color=np.array(c) / 255, label=n) for c, n in zip(colors[:-1], categories)]
            patches.append(mpatches.Patch(color="black", label="ignore"))
            fig.legend(handles=patches, loc="outside lower center", ncol=4, fontsize=8)
            out_path = vis_out / f"cross_eval_{img_path.stem}.png"
            fig.savefig(out_path, dpi=120)
            plt.close(fig)
            print(f"saved visualization: {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/best_decoder_cityscapes.pt")
    ap.add_argument("--cityscapes-root", default="data/cityscapes")
    ap.add_argument("--camvid-root", default="data/camvid")
    ap.add_argument("--category-map", default="data/category_map.json")
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--height", type=int, default=512)
    ap.add_argument("--camvid-resolution", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--out-json", default="outputs/results_phase2.json")
    ap.add_argument("--vis-out", default="outputs/figures")
    ap.add_argument("--num-vis", type=int, default=6)
    ap.add_argument("--wandb", dest="wandb", action="store_true", default=True, help="Weights & Biases logging (default on)")
    ap.add_argument("--no-wandb", dest="wandb", action="store_false")
    ap.add_argument("--wandb-project", default="camvid-seg-phase2")
    ap.add_argument("--wandb-run-id", default=None, help="if set, resume logging onto that training run; otherwise create a new eval run")
    ap.add_argument("--wandb-run-name", default=None)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    with open(Path(args.cityscapes_root) / "class_map.json") as f:
        num_classes = len(json.load(f)["classes"])
    with open(args.category_map) as f:
        cat_meta = json.load(f)

    model = load_model(args.ckpt, num_classes, args.height, args.width, device)

    cs_miou, cs_per_class, cs_n = eval_cityscapes_native(model, args, device)
    print(f"[Cityscapes val, {cs_n} images, 19-class] mIoU = {cs_miou:.4f}")
    for c, v in cs_per_class.items():
        print(f"  {c:15s}: {v:.4f}")

    cross_miou, cross_per_cat, cross_n = eval_camvid_cross(model, args, device, cat_meta)
    print(f"\n[CamVid cross-eval, {cross_n} images, 7-category] mIoU = {cross_miou:.4f}")
    for c, v in cross_per_cat.items():
        print(f"  {c:15s}: {v:.4f}")

    results = {
        "cityscapes_val": {"mIoU": round(cs_miou, 4), "per_class_IoU": cs_per_class, "num_samples": cs_n},
        "camvid_cross_dataset": {"mIoU": round(cross_miou, 4), "per_category_IoU": cross_per_cat, "num_samples": cross_n},
        "note": "camvid_cross_dataset is a fully held-out eval over all 701 images (train+val+test), none of which were used to train this decoder",
    }
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nsaved results: {args.out_json}")

    if args.wandb:
        import wandb

        if args.wandb_run_id:
            run = wandb.init(project=args.wandb_project, id=args.wandb_run_id, resume="must")
        else:
            run = wandb.init(project=args.wandb_project, name=args.wandb_run_name, job_type="eval", config=vars(args))

        log_dict = {
            "test/cityscapes_val_mIoU": cs_miou,
            "test/zero_shot_mIoU": cross_miou,
        }
        for c, v in cs_per_class.items():
            log_dict[f"test/cityscapes_val_iou/{c}"] = v
        for c, v in cross_per_cat.items():
            log_dict[f"test/zero_shot_iou/{c}"] = v
        wandb.log(log_dict)
        wandb.summary["test/cityscapes_val_mIoU"] = cs_miou
        wandb.summary["test/zero_shot_mIoU"] = cross_miou
        wandb.finish()

    visualize_cross_samples(model, args, device, cat_meta, num_vis=args.num_vis)


if __name__ == "__main__":
    main()
