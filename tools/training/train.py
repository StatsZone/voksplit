# tools/training/train.py
# Boucle d'entraînement professionnelle pour VoksplitNet sur MUSDB18-HQ

import os
import time
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model import VoksplitNet
from dataset import MUSDB18Dataset

def compute_loss(pred_voc_mask, pred_ins_mask, mix_mag, target_voc_mag, target_ins_mag):
    """
    Fonction de perte combinée :
    - L1 sur la magnitude estimée (Voix & Instru)
    - Perte Log-Magnitude pour préserver les harmoniques fines
    - Contrainte de conservation d'énergie (somme des masques)
    """
    est_voc_mag = pred_voc_mask * mix_mag
    est_ins_mag = pred_ins_mask * mix_mag

    # L1 Loss
    l1_voc = nn.functional.l1_loss(est_voc_mag, target_voc_mag)
    l1_ins = nn.functional.l1_loss(est_ins_mag, target_ins_mag)

    # Log-magnitude Loss
    eps = 1e-6
    log_voc = nn.functional.l1_loss(torch.log(est_voc_mag + eps), torch.log(target_voc_mag + eps))
    log_ins = nn.functional.l1_loss(torch.log(est_ins_mag + eps), torch.log(target_ins_mag + eps))

    # Mask sum conservation
    mask_sum_loss = nn.functional.mse_loss(pred_voc_mask + pred_ins_mask, torch.ones_like(pred_voc_mask))

    total_loss = (l1_voc + l1_ins) + 0.1 * (log_voc + log_ins) + 0.05 * mask_sum_loss
    return total_loss, l1_voc.item(), l1_ins.item()

def train_epoch(model, dataloader, optimizer, device, epoch):
    model.train()
    total_loss = 0.0
    total_voc_l1 = 0.0
    total_ins_l1 = 0.0
    start_time = time.time()

    for batch_idx, (mix_mag, voc_mag, ins_mag) in enumerate(dataloader):
        mix_mag = mix_mag.to(device)
        voc_mag = voc_mag.to(device)
        ins_mag = ins_mag.to(device)

        optimizer.zero_grad()
        mask_voc, mask_ins = model(mix_mag)

        loss, voc_l1, ins_l1 = compute_loss(mask_voc, mask_ins, mix_mag, voc_mag, ins_mag)
        loss.backward()

        # Gradient clipping pour stabilité
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss += loss.item()
        total_voc_l1 += voc_l1
        total_ins_l1 += ins_l1

        if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(dataloader):
            avg_batch_loss = total_loss / (batch_idx + 1)
            print(f"  [Époque {epoch}] Batch {batch_idx + 1}/{len(dataloader)} | Perte : {avg_batch_loss:.4f} (Voc: {voc_l1:.4f}, Ins: {ins_l1:.4f})", end="\r")

    elapsed = time.time() - start_time
    print(f"\n  -> Époque {epoch} terminée en {elapsed:.1f}s | Perte moyenne : {total_loss / len(dataloader):.4f}")
    return total_loss / len(dataloader)

def validate(model, dataloader, device):
    model.eval()
    val_loss = 0.0
    val_sdr_approx = 0.0
    count = 0

    with torch.no_grad():
        for mix_mag, voc_mag, ins_mag in dataloader:
            mix_mag = mix_mag.to(device)
            voc_mag = voc_mag.to(device)
            ins_mag = ins_mag.to(device)

            mask_voc, mask_ins = model(mix_mag)
            loss, _, _ = compute_loss(mask_voc, mask_ins, mix_mag, voc_mag, ins_mag)
            val_loss += loss.item()

            # SDR approximation en dB (Signal-to-Distortion Ratio basé sur l'énergie résiduelle)
            est_voc = mask_voc * mix_mag
            signal_power = torch.sum(voc_mag ** 2)
            noise_power = torch.sum((voc_mag - est_voc) ** 2) + 1e-7
            sdr = 10 * torch.log10(signal_power / noise_power)
            val_sdr_approx += sdr.item()
            count += 1

    return val_loss / len(dataloader), val_sdr_approx / max(1, count)

def main():
    parser = argparse.ArgumentParser(description="Entraînement de VoksplitNet pour la séparation Voix/Instru")
    parser.add_argument("--data_dir", type=str, default="./data/musdb18hq", help="Chemin vers le dataset MUSDB18-HQ")
    parser.add_argument("--epochs", type=int, default=50, help="Nombre total d'époques")
    parser.add_argument("--batch_size", type=int, default=8, help="Taille des batchs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Taux d'apprentissage initial")
    parser.add_argument("--output_dir", type=str, default="./checkpoints", help="Dossier de sauvegarde des modèles")
    parser.add_argument("--resume", type=str, default=None, help="Chemin d'un checkpoint à reprendre")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Entraînement VoksplitNet ===")
    print(f"Périphérique : {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    # Initialisation modèle & compteurs de paramètres
    model = VoksplitNet(in_channels=2, base_channels=32, num_freq_bins=1024).to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Paramètres du modèle : {num_params:,} ({num_params * 4 / (1024*1024):.2f} Mo en FP32, ~{num_params * 2 / (1024*1024):.2f} Mo en FP16)")

    # Datasets & Dataloaders
    train_dataset = MUSDB18Dataset(args.data_dir, split="train", segment_seconds=6.0, augment=True)
    val_dataset = MUSDB18Dataset(args.data_dir, split="test", segment_seconds=6.0, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=(device.type == "cuda"))
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=1)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    start_epoch = 1
    best_sdr = -999.0

    if args.resume and os.path.exists(args.resume):
        print(f"Reprise depuis le checkpoint : {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint.get("epoch", 1) + 1
        best_sdr = checkpoint.get("best_sdr", -999.0)

    for epoch in range(start_epoch, args.epochs + 1):
        print(f"\n--- Époque {epoch}/{args.epochs} (LR: {scheduler.get_last_lr()[0]:.6f}) ---")
        train_loss = train_epoch(model, train_loader, optimizer, device, epoch)
        val_loss, val_sdr = validate(model, val_loader, device)

        print(f"Validation : Perte = {val_loss:.4f} | SDR Vocaux estimé = {val_sdr:.2f} dB (Meilleur: {best_sdr:.2f} dB)")
        scheduler.step()

        # Sauvegarde du meilleur modèle
        if val_sdr > best_sdr:
            best_sdr = val_sdr
            best_path = os.path.join(args.output_dir, "best_voksplit_model.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_sdr": best_sdr,
            }, best_path)
            print(f"  🏆 Nouveau record ! Modèle sauvegardé dans : {best_path}")

        # Sauvegarde périodique tous les 10 époques
        if epoch % 10 == 0:
            ckpt_path = os.path.join(args.output_dir, f"checkpoint_epoch_{epoch}.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "best_sdr": best_sdr
            }, ckpt_path)

    print("\n✅ Entraînement terminé avec succès !")

if __name__ == "__main__":
    main()
