import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn

# Local imports
from preprocessing.dataloader import get_dataloaders
from train import train
from metrics.loss import DiceLoss, CombinedLoss
from models.deeplabv3plus import DeepLabV3Plus
from metrics.metrics import dice_score, iou_score


# Aquí ponemos el modelo que queramos usar
MODEL_TYPE = "deeplabv3plus"

# Tonterías varias
BATCH_SIZE = 16
NUM_EPOCHS = 50
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
IMAGE_SIZE = 256

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_DIR = "outputs/experiments"

def main():
    print(f"--- Starting Segmentation Training Pipeline ---")
    print(f"Configuration: Model: {MODEL_TYPE}, Batch Size: {BATCH_SIZE}, "
          f"Epochs: {NUM_EPOCHS}, LR: {LEARNING_RATE}")

    device = torch.device(DEVICE)

    print("\n--- Setup Dataloaders ---")
    dataloaders = get_dataloaders(
        batch_size=BATCH_SIZE,
        image_size=(IMAGE_SIZE, IMAGE_SIZE)
    )

    print("\n--- Setup Architecture ---")
    if MODEL_TYPE == "deeplabv3plus":
        print("[Model] Initializing CNN-based model (DeepLabV3+).")
        model = DeepLabV3Plus(num_classes=1).to(device)
    else:
        # Placeholder for other models (Transformer, etc.)
        # Modify this section to instantiate other model classes as needed.
        raise ValueError(f"Model {MODEL_TYPE} not yet configured in pipeline.py.")

    print("\n--- Setup Optimization ---")
    # Weighted combination of BCE and Dice Loss for optimal segmentation performance
    criterion = CombinedLoss(bce_weight=0.5, dice_weight=0.5).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    print("\n--- Begin Training ---")
    history = train(
        model=model,
        dataloaders=dataloaders,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        num_epochs=NUM_EPOCHS,
        checkpoint_every=True,
        save_dir=os.path.join(SAVE_DIR, MODEL_TYPE)
    )
    
    print("\n--- Training Complete ---")
    best_dice = max(history["val_dice"])
    best_iou = max(history["val_iou"])
    print(f"Best Validation Dice: {best_dice:.4f}")
    print(f"Best Validation IoU:  {best_iou:.4f}")

if __name__ == "__main__":
    main()
