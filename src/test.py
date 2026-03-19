import torch
import torch.nn as nn
import pandas as pd
import os


@torch.no_grad()
def epoch_test(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_dice = 0.0
    running_iou = 0.0
    
    for batch_idx, (images, masks) in enumerate(dataloader):
        images = images.to(device)
        masks = masks.to(device)

        outputs = model(images)
        loss = criterion(outputs, masks)
        
        running_loss += loss.item()
        running_dice += dice_score(outputs, masks, apply_sigmoid=True).item()
        running_iou += iou_score(outputs, masks, apply_sigmoid=True).item()

    epoch_loss = running_loss / len(dataloader)
    epoch_dice = running_dice / len(dataloader)
    epoch_iou = running_iou / len(dataloader)
    return epoch_loss, epoch_dice, epoch_iou


def test(model, dataloaders, criterion, device, save_dir="outputs"):
    os.makedirs(save_dir, exist_ok=True)

    test_loader = dataloaders["test"]
    test_loss, test_dice, test_iou = epoch_test(model, test_loader, criterion, device)

    # Store and save metrics
    results = {
        "test_loss": [test_loss],
        "test_dice": [test_dice],
        "test_iou":  [test_iou],
    }
    pd.DataFrame(results).to_csv(os.path.join(save_dir, "test_metrics.csv"), index=False)

    print(f"Test Loss: {test_loss:.4f} | Dice: {test_dice:.4f} | IoU: {test_iou:.4f}")

    return results

