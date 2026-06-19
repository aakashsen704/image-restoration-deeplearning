import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os, time, sys
from tqdm import tqdm

sys.path.append("/content")
from dataset import ImageRestorationDataset
from unet import get_model
from metrics import psnr, ssim

CONFIG = {
    "train_blur":    "data/train/blur",
    "train_sharp":   "data/train/sharp",
    "val_blur":      "data/val/blur",
    "val_sharp":     "data/val/sharp",
    "batch_size":    8,
    "num_epochs":    100,
    "learning_rate": 2e-4,
    "patch_size":    256,
    "save_dir":      "checkpoints",
}

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using: {device}")
    os.makedirs(CONFIG["save_dir"], exist_ok=True)

    train_loader = DataLoader(
        ImageRestorationDataset(CONFIG["train_blur"], CONFIG["train_sharp"], is_train=True),
        batch_size=CONFIG["batch_size"], shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(
        ImageRestorationDataset(CONFIG["val_blur"], CONFIG["val_sharp"], is_train=False),
        batch_size=1, shuffle=False, num_workers=2)

    model     = get_model(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["learning_rate"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG["num_epochs"], eta_min=1e-6)
    l1_loss   = nn.L1Loss()
    best_psnr = 0.0

    for epoch in range(1, CONFIG["num_epochs"] + 1):
        model.train()
        train_loss = 0.0
        for blur, sharp in tqdm(train_loader, desc=f"Epoch {epoch}"):
            blur, sharp = blur.to(device), sharp.to(device)
            optimizer.zero_grad()
            output = model(blur)
            loss = l1_loss(output, sharp)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
        scheduler.step()

        model.eval()
        val_psnr, val_ssim = 0.0, 0.0
        with torch.no_grad():
            for blur, sharp in val_loader:
                blur, sharp = blur.to(device), sharp.to(device)
                output = torch.clamp(model(blur), 0, 1)
                val_psnr += psnr(output, sharp).item()
                val_ssim += ssim(output, sharp).item()

        avg_psnr = val_psnr / len(val_loader)
        avg_ssim = val_ssim / len(val_loader)
        print(f"Epoch {epoch} | Loss: {train_loss/len(train_loader):.4f} | PSNR: {avg_psnr:.2f}dB | SSIM: {avg_ssim:.4f}")

        if avg_psnr > best_psnr:
            best_psnr = avg_psnr
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                        "psnr": avg_psnr, "ssim": avg_ssim},
                       f"{CONFIG['save_dir']}/best_model.pth")
            print(f"  ✅ Best model saved! PSNR: {best_psnr:.2f}dB")

if __name__ == "__main__":
    train()
