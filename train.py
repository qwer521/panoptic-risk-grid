import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
from dataset import DashcamBEVDataset
from model import BEVNetResNet50, BEVNetResNet101
import matplotlib.pyplot as plt
from torch import amp, autocast
from tqdm import tqdm
import re



# Config
EPOCHS = 100
BATCH_SIZE = 16
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
scaler = amp.GradScaler("cuda")

# Dice loss function
def dice_loss(pred, target, smooth=1e-6):
    pred = torch.sigmoid(pred)  # ensure it's between 0-1
    pred_flat = pred.view(pred.size(0), -1)
    target_flat = target.view(target.size(0), -1)
    intersection = (pred_flat * target_flat).sum(1)
    union = pred_flat.sum(1) + target_flat.sum(1)
    dice = (2. * intersection + smooth) / (union + smooth)
    return 1 - dice.mean()

# Run folder
base_dir = "models"
os.makedirs(base_dir, exist_ok=True)

existing_runs = [
    int(match.group(1)) for name in os.listdir(base_dir)
    if (match := re.match(r"run(\d+)", name))
]
run_number = max(existing_runs, default=0) + 1
model_dir = os.path.join(base_dir, f"run{run_number}")
os.makedirs(model_dir)
print(f"🔹 Saving model to: {model_dir}")

# Load dataset
dataset = DashcamBEVDataset("images", "panoptic", "labels")
val_ratio = 0.2
val_size = int(len(dataset) * val_ratio)
train_size = len(dataset) - val_size

train_set, val_set = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)


# Init model
model = BEVNetResNet50().to(DEVICE)
pos_weight = torch.tensor([3.0]).to(DEVICE)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

loss_history = []
val_loss_history = []

# Training loop
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0
    for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]", leave=False):
        inputs = inputs.to(DEVICE)
        targets = targets.to(DEVICE)

        with autocast(device_type="cuda"):
            outputs = model(inputs)
            bce_loss = criterion(outputs, targets)
            dice = dice_loss(outputs, targets)
            loss = 0.5 * bce_loss + 0.5 * dice

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()


        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    loss_history.append(avg_loss)
    print(f"[Epoch {epoch+1}/{EPOCHS}] Loss: {avg_loss:.4f}")

    scheduler.step()

    # ─────── Validation Phase ───────
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, targets in tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]", leave=False):
            inputs = inputs.to(DEVICE)
            targets = targets.to(DEVICE)

            with autocast(device_type="cuda"):
                outputs = model(inputs)
                bce_loss = criterion(outputs, targets)
                dice = dice_loss(outputs, targets)
                loss = 0.5 * bce_loss + 0.5 * dice

            val_loss += loss.item()

    val_avg = val_loss / len(val_loader)
    val_loss_history.append(val_avg)
    print(f"Validation Loss: {val_avg:.4f}")

    # Save every 20 epochs
    if (epoch + 1) % 20 == 0:
        save_path = os.path.join(model_dir, f"epoch_{epoch+1}.pth")
        torch.save(model.state_dict(), save_path)
        print(f"Model saved to: {save_path}")

        plt.clf()
        plt.plot(range(1, len(loss_history)+1), loss_history, label="Train")
        plt.plot(range(1, len(val_loss_history)+1), val_loss_history, label="Validation")
        plt.title("Training Loss Over Epochs")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(model_dir, "training_loss.png"))

# Save model
final_path = os.path.join(model_dir, f"last.pth")
torch.save(model.state_dict(), final_path)
print(f"Final model saved to: {final_path}")

plt.clf()
plt.plot(range(1, EPOCHS+1), loss_history, label="Train")
plt.plot(range(1, EPOCHS+1), val_loss_history, label="Validation")
plt.title("Training Loss Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(model_dir, "training_loss.png"))