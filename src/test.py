import torch
import torch.nn as nn
import pandas as pd
import os
import matplotlib.pyplot as plt


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

    metrics_csv_path = os.path.join(save_dir, "metrics.csv")
    if os.path.exists(metrics_csv_path):
        df = pd.read_csv(metrics_csv_path)
        if not df.empty and "val_dice" in df.columns:
            # Save the best results apart
            best_idx = df["val_dice"].idxmax()
            best_results = df.iloc[[best_idx]]
            best_results.to_csv(os.path.join(save_dir, "best_metrics.csv"), index=False)

            # Extraer las epoch para los plots
            epochs = df["epoch"]

            # Plot Loss vs Epochs
            plt.figure()
            plt.plot(epochs, df["train_loss"], label="Train Loss")
            plt.plot(epochs, df["val_loss"], label="Val Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend()
            plt.title("Loss vs Epochs")
            plt.savefig(os.path.join(save_dir, "loss_vs_epochs.png"))
            plt.close()

            # Plot Dice vs Epochs
            plt.figure()
            plt.plot(epochs, df["train_dice"], label="Train Dice")
            plt.plot(epochs, df["val_dice"], label="Val Dice")
            plt.xlabel("Epoch")
            plt.ylabel("Dice")
            plt.legend()
            plt.title("Dice vs Epochs")
            plt.savefig(os.path.join(save_dir, "dice_vs_epochs.png"))
            plt.close()

            # Plot IoU vs Epochs
            plt.figure()
            plt.plot(epochs, df["train_iou"], label="Train IoU")
            plt.plot(epochs, df["val_iou"], label="Val IoU")
            plt.xlabel("Epoch")
            plt.ylabel("IoU")
            plt.legend()
            plt.title("IoU vs Epochs")
            plt.savefig(os.path.join(save_dir, "iou_vs_epochs.png"))
            plt.close()

    # Save 3 Real vs Predicted Mask Comparisons
    pred_dir = os.path.join(save_dir, "predicted_masks")
    os.makedirs(pred_dir, exist_ok=True)
    
    model.eval()
    num_saved = 0
    with torch.no_grad():
        for images, masks in test_loader:
            images = images.to(device)
            outputs = model(images)
            preds = (torch.sigmoid(outputs) > 0.5).float()
            
            for i in range(images.size(0)):
                if num_saved >= 3:
                    break
                
                img = images[i].cpu()
                if img.shape[0] == 3:
                    img = img.permute(1, 2, 0).numpy()
                elif img.shape[0] == 1:
                    img = img.squeeze(0).numpy()
                else:
                    img = img.numpy()
                
                mask = masks[i].cpu().squeeze().numpy()
                pred = preds[i].cpu().squeeze().numpy()
                
                plt.figure(figsize=(15, 5))
                plt.subplot(1, 3, 1)
                if len(img.shape) == 2:
                    plt.imshow(img, cmap="gray")
                else:
                    plt.imshow(img)
                plt.title("Image")
                plt.axis("off")
                
                plt.subplot(1, 3, 2)
                plt.imshow(mask, cmap="gray")
                plt.title("Real Mask")
                plt.axis("off")
                
                plt.subplot(1, 3, 3)
                plt.imshow(pred, cmap="gray")
                plt.title("Predicted Mask")
                plt.axis("off")
                
                plt.savefig(os.path.join(pred_dir, f"comparison_{num_saved}.png"), bbox_inches="tight")
                plt.close()
                num_saved += 1
            
            if num_saved >= 3:
                break

    return results
