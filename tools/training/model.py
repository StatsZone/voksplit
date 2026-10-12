# tools/training/model.py
# Architecture PyTorch de VoksplitNet : TFC-TDF U-Net pour la séparation Voix / Instrumental
# Ultra-compact (~8M paramètres, ~18-35 Mo en FP16), optimisé pour l'inférence WebGPU / WASM

import torch
import torch.nn as nn
import torch.nn.functional as F

class TFC(nn.Module):
    """Time-Frequency Convolution block with residual connection"""
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.act1 = nn.GELU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act2 = nn.GELU()
        self.shortcut = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        res = self.shortcut(x)
        x = self.act1(self.bn1(self.conv1(x)))
        x = self.act2(self.bn2(self.conv2(x)))
        return x + res

class TDF(nn.Module):
    """Time-Distributed Fully connected layer along frequency bins"""
    def __init__(self, channels, num_freq_bins, bottleneck_ratio=4):
        super().__init__()
        bottleneck = max(8, num_freq_bins // bottleneck_ratio)
        self.linear1 = nn.Linear(num_freq_bins, bottleneck)
        self.act = nn.GELU()
        self.linear2 = nn.Linear(bottleneck, num_freq_bins)
        self.norm = nn.LayerNorm(num_freq_bins)

    def forward(self, x):
        # x shape: [Batch, Channels, Freq, Time]
        b, c, f, t = x.shape
        x_trans = x.permute(0, 1, 3, 2) # [B, C, T, F]
        res = x_trans
        out = self.linear2(self.act(self.linear1(x_trans)))
        out = self.norm(out + res)
        return out.permute(0, 1, 3, 2) # [B, C, F, T]

class TFC_TDF_Block(nn.Module):
    def __init__(self, in_channels, out_channels, num_freq_bins):
        super().__init__()
        self.tfc = TFC(in_channels, out_channels)
        self.tdf = TDF(out_channels, num_freq_bins)

    def forward(self, x):
        return self.tdf(self.tfc(x))

class VoksplitNet(nn.Module):
    """
    Réseau U-Net multi-échelles TFC-TDF pour la séparation stéréo Voix vs Instrumental.
    Entrée : Spectrogramme stéréo magnitude [B, 2, F=1024, T]
    Sortie : Masques souples pour Voix [B, 2, F, T] et Instrumental [B, 2, F, T] (plage [0, 1])
    """
    def __init__(self, in_channels=2, base_channels=32, num_freq_bins=1024):
        super().__init__()
        self.num_freq_bins = num_freq_bins

        # Encodeur
        self.enc1 = TFC_TDF_Block(in_channels, base_channels, num_freq_bins)
        self.down1 = nn.MaxPool2d(kernel_size=(2, 2))
        f_down1 = num_freq_bins // 2  # 512

        self.enc2 = TFC_TDF_Block(base_channels, base_channels * 2, f_down1)
        self.down2 = nn.MaxPool2d(kernel_size=(2, 2))
        f_down2 = f_down1 // 2        # 256

        self.enc3 = TFC_TDF_Block(base_channels * 2, base_channels * 4, f_down2)
        self.down3 = nn.MaxPool2d(kernel_size=(2, 2))
        f_down3 = f_down2 // 2        # 128

        # Bottleneck
        self.bottleneck = TFC_TDF_Block(base_channels * 4, base_channels * 8, f_down3)

        # Décodeur
        self.up3 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, kernel_size=(2, 2), stride=(2, 2))
        self.dec3 = TFC_TDF_Block(base_channels * 8, base_channels * 4, f_down2)

        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=(2, 2), stride=(2, 2))
        self.dec2 = TFC_TDF_Block(base_channels * 4, base_channels * 2, f_down1)

        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=(2, 2), stride=(2, 2))
        self.dec1 = TFC_TDF_Block(base_channels * 2, base_channels, num_freq_bins)

        # Têtes de sortie (Stéréo : 2 canaux Voix + 2 canaux Instrumental)
        self.head_vocals = nn.Sequential(
            nn.Conv2d(base_channels, 2, kernel_size=1),
            nn.Sigmoid()
        )
        self.head_instrumental = nn.Sequential(
            nn.Conv2d(base_channels, 2, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        orig_t = x.shape[3]

        # Padding en temps si besoin pour être multiple de 8 (3 niveaux de pooling x2)
        pad_t = (8 - (orig_t % 8)) % 8
        if pad_t > 0:
            x = F.pad(x, (0, pad_t, 0, 0))

        # Pass Encodeur
        e1 = self.enc1(x)
        e2 = self.enc2(self.down1(e1))
        e3 = self.enc3(self.down2(e2))

        # Bottleneck
        b = self.bottleneck(self.down3(e3))

        # Pass Décodeur avec Skip-connections
        d3 = self.up3(b)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        mask_vocals = self.head_vocals(d1)[:, :, :, :orig_t]
        mask_instru = self.head_instrumental(d1)[:, :, :, :orig_t]

        return mask_vocals, mask_instru
