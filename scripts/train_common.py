"""Dataset-agnostic training pieces: CE+Dice loss, eval loop. Shared by train.py / train_cityscapes.py."""
import torch
import torch.nn as nn
import torch.nn.functional as F


def dice_loss(logits: torch.Tensor, target: torch.Tensor, ignore_index: int, num_classes: int, eps: float = 1e-6):
    probs = F.softmax(logits, dim=1)
    valid = (target != ignore_index)
    target_safe = target.clone()
    target_safe[~valid] = 0  # placeholder to satisfy one_hot, masked out below
    target_onehot = F.one_hot(target_safe, num_classes).permute(0, 3, 1, 2).float()
    valid = valid.unsqueeze(1).float()
    probs = probs * valid
    target_onehot = target_onehot * valid
    dims = (0, 2, 3)
    intersection = (probs * target_onehot).sum(dims)
    union = probs.sum(dims) + target_onehot.sum(dims)
    dice_per_class = (2 * intersection + eps) / (union + eps)
    return 1 - dice_per_class.mean()


class CEDiceLoss(nn.Module):
    def __init__(self, num_classes: int, ignore_index: int, dice_weight: float = 1.0):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.ignore_index = ignore_index
        self.num_classes = num_classes
        self.dice_weight = dice_weight

    def forward(self, logits, target):
        ce = self.ce(logits, target)
        dice = dice_loss(logits, target, self.ignore_index, self.num_classes)
        return ce + self.dice_weight * dice, ce.item(), dice.item()


def evaluate(model, loader, criterion, jaccard, device, jaccard_per_class=None):
    """Passing jaccard_per_class (optional, a MulticlassJaccardIndex with average=None) also returns
    a per-class IoU list, for logging per-class curves to W&B. None if not passed."""
    model.decoder.eval()
    jaccard.reset()
    if jaccard_per_class is not None:
        jaccard_per_class.reset()
    total_loss = 0.0
    n = 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs)
            loss, _, _ = criterion(logits, labels)
            total_loss += loss.item() * imgs.size(0)
            n += imgs.size(0)
            preds = logits.argmax(dim=1)
            jaccard.update(preds, labels)
            if jaccard_per_class is not None:
                jaccard_per_class.update(preds, labels)
    model.decoder.train()
    per_class = jaccard_per_class.compute().tolist() if jaccard_per_class is not None else None
    return total_loss / n, jaccard.compute().item(), per_class


def log_prediction_panel(model, dataset, device, num_samples: int = 4):
    """Build a list of wandb.Image for a W&B interactive segmentation mask overlay (prediction/GT
    toggle). dataset must expose classes/ignore_index/mean/std like CamVidDataset/CityscapesDataset."""
    import wandb

    class_labels = {i: name for i, name in enumerate(dataset.classes)}
    class_labels[dataset.ignore_index] = "ignore"

    model.decoder.eval()
    images = []
    step = max(len(dataset) // num_samples, 1)
    with torch.no_grad():
        for i in range(0, len(dataset), step):
            if len(images) >= num_samples:
                break
            img_t, label_t = dataset[i]
            logits = model(img_t.unsqueeze(0).to(device))
            pred = logits.argmax(dim=1).squeeze(0).cpu().numpy()
            gt = label_t.numpy()
            img_np = (img_t * dataset.std + dataset.mean).clamp(0, 1).permute(1, 2, 0).numpy()
            images.append(
                wandb.Image(
                    img_np,
                    masks={
                        "prediction": {"mask_data": pred, "class_labels": class_labels},
                        "ground_truth": {"mask_data": gt, "class_labels": class_labels},
                    },
                )
            )
    model.decoder.train()
    return images
