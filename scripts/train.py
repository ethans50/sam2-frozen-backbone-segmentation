"""Train a SAM2 backbone (frozen) + decoder. Loss = CrossEntropy + Dice (see DETAILS.md).
"""
import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))

from dataset import CamVidDataset  # noqa: E402
from model import SAM2Backbone, SegModel  # noqa: E402
from train_common import CEDiceLoss, evaluate, log_prediction_panel  # noqa: E402


class AugCamVidDataset(CamVidDataset):
    """Train-split only: random horizontal flip + random crop. val uses no augmentation."""

    def __init__(self, *args, crop_scale: float = 1.15, **kwargs):
        super().__init__(*args, **kwargs)
        self.crop_scale = crop_scale

    def __getitem__(self, i):
        img_path = self.img_paths[i]
        label_path = self.label_dir / img_path.name
        if not label_path.exists():
            raise FileNotFoundError(f"decoded label not found: {label_path}")

        from PIL import Image

        resolution = self.size_wh[0]
        big = int(round(resolution * self.crop_scale))
        img = Image.open(img_path).convert("RGB").resize((big, big), Image.BILINEAR)
        label = Image.open(label_path).resize((big, big), Image.NEAREST)

        max_off = big - resolution
        left = random.randint(0, max_off)
        top = random.randint(0, max_off)
        box = (left, top, left + resolution, top + resolution)
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
    ap.add_argument("--data-root", default="data/camvid")
    ap.add_argument("--resolution", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=24)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=8, help="early-stop patience when val mIoU stops improving")
    ap.add_argument("--ckpt-out", default="checkpoints/best_decoder.pt")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--wandb", dest="wandb", action="store_true", default=True, help="Weights & Biases logging (default on)")
    ap.add_argument("--no-wandb", dest="wandb", action="store_false")
    ap.add_argument("--wandb-project", default="camvid-seg-phase1")
    ap.add_argument("--wandb-run-name", default=None)
    ap.add_argument("--wandb-image-every", type=int, default=10, help="log prediction image panel every N epochs")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_ds = AugCamVidDataset("train", data_root=args.data_root, resolution=args.resolution)
    val_ds = CamVidDataset("val", data_root=args.data_root, resolution=args.resolution)
    num_classes = len(train_ds.classes)
    ignore_index = train_ds.ignore_index

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    backbone = SAM2Backbone(device=device)
    with torch.no_grad():
        dummy = torch.randn(1, 3, args.resolution, args.resolution, device=device)
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
