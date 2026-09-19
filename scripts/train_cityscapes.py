"""Train a SAM2(large) backbone (frozen) + decoder on Cityscapes (19-class trainId).
Loss = CrossEntropy + Dice. Model selection uses Cityscapes val only — CamVid is never
looked at (see DETAILS.md Phase 2 "학습/평가 전략"; cross-eval lives in eval_phase2.py).
"""
import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))

from dataset import CityscapesDataset  # noqa: E402
from model import SAM2Backbone, SegModel  # noqa: E402
from train_common import CEDiceLoss, evaluate, log_prediction_panel  # noqa: E402


class AugCityscapesDataset(CityscapesDataset):
    """Train-split only: random horizontal flip + random crop (upscale then crop, keeping 2:1 aspect ratio)."""

    def __init__(self, *args, crop_scale: float = 1.15, **kwargs):
        super().__init__(*args, **kwargs)
        self.crop_scale = crop_scale

    def __getitem__(self, i):
        img_path = self.img_paths[i]
        stem = img_path.name.replace("_leftImg8bit.png", "")
        label_path = self.label_dir / f"{stem}.png"
        if not label_path.exists():
            raise FileNotFoundError(f"decoded label not found: {label_path}")

        w, h = self.size_wh
        big_w, big_h = int(round(w * self.crop_scale)), int(round(h * self.crop_scale))
        img = Image.open(img_path).convert("RGB").resize((big_w, big_h), Image.BILINEAR)
        label = Image.open(label_path).resize((big_w, big_h), Image.NEAREST)

        left = random.randint(0, big_w - w)
        top = random.randint(0, big_h - h)
        box = (left, top, left + w, top + h)
        img = img.crop(box)
        label = label.crop(box)

        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            label = label.transpose(Image.FLIP_LEFT_RIGHT)

        img_t = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
        img_t = (img_t - self.mean) / self.std
        label_t = torch.from_numpy(np.array(label)).long()
        return img_t, label_t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data/cityscapes")
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--height", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=12, help="early-stop patience when val mIoU stops improving")
    ap.add_argument("--ckpt-out", default="checkpoints/best_decoder_cityscapes.pt")
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--wandb", dest="wandb", action="store_true", default=True, help="Weights & Biases logging (default on)")
    ap.add_argument("--no-wandb", dest="wandb", action="store_false")
    ap.add_argument("--wandb-project", default="camvid-seg-phase2")
    ap.add_argument("--wandb-run-name", default=None)
    ap.add_argument("--wandb-image-every", type=int, default=10, help="log prediction image panel every N epochs")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    size_wh = (args.width, args.height)

    train_ds = AugCityscapesDataset("train", data_root=args.data_root, size_wh=size_wh)
    val_ds = CityscapesDataset("val", data_root=args.data_root, size_wh=size_wh)
    num_classes = len(train_ds.classes)
    ignore_index = train_ds.ignore_index

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, drop_last=True, persistent_workers=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=max(args.num_workers // 2, 1), persistent_workers=True,
    )

    backbone = SAM2Backbone(device=device, variant="large")
    with torch.no_grad():
        dummy = torch.randn(1, 3, args.height, args.width, device=device)
        channels = [f.shape[1] for f in backbone(dummy)]
    model = SegModel(num_classes=num_classes, backbone=backbone, backbone_channels=channels).to(device)

    criterion = CEDiceLoss(num_classes=num_classes, ignore_index=ignore_index)
    optimizer = torch.optim.AdamW(model.decoder.parameters(), lr=args.lr)

    from torchmetrics.classification import MulticlassJaccardIndex
    jaccard = MulticlassJaccardIndex(num_classes=num_classes, ignore_index=ignore_index).to(device)
    jaccard_per_class = MulticlassJaccardIndex(num_classes=num_classes, ignore_index=ignore_index, average=None).to(device)

    run = None
    if args.wandb:
        import wandb

        run = wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=vars(args))

    best_miou = -1.0
    best_epoch = -1
    epochs_since_improve = 0
    Path(args.ckpt_out).parent.mkdir(parents=True, exist_ok=True)

    epoch_bar = tqdm(range(1, args.epochs + 1), desc="epochs", unit="epoch")
    for epoch in epoch_bar:
        epoch_start = time.time()
        model.decoder.train()
        running_loss = running_ce = running_dice = 0.0
        n = 0
        pbar = tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}", leave=False)
        for imgs, labels in pbar:
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs)
            loss, ce_val, dice_val = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            bs = imgs.size(0)
            running_loss += loss.item() * bs
            running_ce += ce_val * bs
            running_dice += dice_val * bs
            n += bs
            pbar.set_postfix(loss=f"{running_loss/n:.4f}")
        train_loss = running_loss / n

        val_loss, val_miou, val_iou_per_class = evaluate(
            model, val_loader, criterion, jaccard, device, jaccard_per_class=jaccard_per_class
        )
        epoch_time = time.time() - epoch_start
        epoch_bar.set_postfix(val_mIoU=f"{val_miou:.4f}", epoch_s=f"{epoch_time:.1f}")
        tqdm.write(
            f"epoch {epoch:3d}/{args.epochs} | train_loss {train_loss:.4f} "
            f"(ce {running_ce/n:.4f} dice {running_dice/n:.4f}) | val_loss {val_loss:.4f} | val_mIoU {val_miou:.4f}"
        )

        if run:
            log_dict = {
                "epoch": epoch,
                "train/loss": train_loss,
                "train/ce_loss": running_ce / n,
                "train/dice_loss": running_dice / n,
                "val/loss": val_loss,
                "val/mIoU": val_miou,
                "epoch_time_sec": epoch_time,
            }
            for cls_name, iou in zip(train_ds.classes, val_iou_per_class):
                log_dict[f"val/iou/{cls_name}"] = iou
            if epoch % args.wandb_image_every == 0 or epoch == args.epochs:
                log_dict["val/predictions"] = log_prediction_panel(model, val_ds, device)
            wandb.log(log_dict, step=epoch)

        if val_miou > best_miou:
            best_miou = val_miou
            best_epoch = epoch
            epochs_since_improve = 0
            torch.save(model.decoder.state_dict(), args.ckpt_out)
            if run:
                wandb.summary["best/val_mIoU"] = best_miou
                wandb.summary["best/epoch"] = best_epoch
        else:
            epochs_since_improve += 1
            if epochs_since_improve >= args.patience:
                tqdm.write(f"early stop: epoch {epoch}, no val_mIoU improvement for {args.patience} epochs")
                break

    epoch_bar.close()
    print(f"\nbest val_mIoU {best_miou:.4f} @ epoch {best_epoch} -> {args.ckpt_out}")

    if run:
        artifact = wandb.Artifact(f"decoder-{run.id}", type="model", metadata={"best_val_mIoU": best_miou, "best_epoch": best_epoch})
        artifact.add_file(args.ckpt_out)
        wandb.log_artifact(artifact)
        wandb.finish()


if __name__ == "__main__":
    main()
