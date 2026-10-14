# tools/training/dataset.py
# Dataset PyTorch 100% Commercial-Compliant (CC0 / CC-BY / Domaine Public)
# Méthode : Dynamic Synthetic Remixing (Acapellas Libres + Instrumentaux Libres)

import os
import glob
import random
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchaudio

class CommercialStemDataset(Dataset):
    """
    Dataset PyTorch 100% Conforme Commercialement pour l'isolation vocale.
    Associe dynamiquement des pistes de voix pures (acapellas libres)
    et des pistes instrumentales pures (productions libres CC0 / CC-BY).
    
    Structure attendue :
      root_dir/
        vocals/ (ou train/vocals/)
          *.wav, *.mp3, *.flac
        instrumentals/ (ou train/instrumentals/)
          *.wav, *.mp3, *.flac
    """
    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        sample_rate: int = 44100,
        segment_seconds: float = 6.0,
        n_fft: int = 2048,
        hop_length: int = 512,
        augment: bool = True
    ):
        super().__init__()
        self.root_dir = root_dir
        self.split = split
        self.sample_rate = sample_rate
        self.segment_samples = int(segment_seconds * sample_rate)
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.augment = augment and (split == "train")
        self.window = torch.hann_window(n_fft)

        # Recherche des dossiers de voix et d'instrumentaux
        voc_dir = os.path.join(root_dir, split, "vocals") if os.path.exists(os.path.join(root_dir, split, "vocals")) else os.path.join(root_dir, "vocals")
        ins_dir = os.path.join(root_dir, split, "instrumentals") if os.path.exists(os.path.join(root_dir, split, "instrumentals")) else os.path.join(root_dir, "instrumentals")

        valid_exts = ('.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac', '.opus', '.aif', '.aiff')
        self.vocal_files = []
        self.instrumental_files = []

        if os.path.exists(voc_dir):
            for root, _, files in os.walk(voc_dir):
                for f in files:
                    if not f.startswith('.') and f.lower().endswith(valid_exts):
                        self.vocal_files.append(os.path.join(root, f))

        if os.path.exists(ins_dir):
            for root, _, files in os.walk(ins_dir):
                for f in files:
                    if not f.startswith('.') and f.lower().endswith(valid_exts):
                        self.instrumental_files.append(os.path.join(root, f))

        self.vocal_files = sorted(self.vocal_files)
        self.instrumental_files = sorted(self.instrumental_files)

        print(f"[{split.upper()}] Pistes Détectées : {len(self.vocal_files)} Acapellas | {len(self.instrumental_files)} Instrumentaux")

    def __len__(self):
        # En mode Remixing Dynamique, N voix x M instrumentaux = grand nombre de combinaisons
        if len(self.vocal_files) == 0 or len(self.instrumental_files) == 0:
            return 100 # Mode démo de secours si aucun fichier
        return max(len(self.vocal_files), len(self.instrumental_files)) * (10 if self.split == "train" else 1)

    def _load_audio_chunk(self, path: str):
        try:
            info = torchaudio.info(path)
            total_samples = info.num_frames
            
            if total_samples <= self.segment_samples:
                waveform, sr = torchaudio.load(path)
                if sr != self.sample_rate:
                    waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)
                # Conversion stéréo si mono
                if waveform.shape[0] == 1:
                    waveform = waveform.repeat(2, 1)
                elif waveform.shape[0] > 2:
                    waveform = waveform[:2, :]
                # Padding si court
                if waveform.shape[-1] < self.segment_samples:
                    waveform = F.pad(waveform, (0, self.segment_samples - waveform.shape[-1]))
                return waveform[:, :self.segment_samples]

            # Découpe aléatoire d'un segment de 6s
            max_offset = max(0, total_samples - self.segment_samples)
            offset = random.randint(0, max_offset) if self.augment else 0

            waveform, sr = torchaudio.load(path, frame_offset=offset, num_frames=self.segment_samples)
            if sr != self.sample_rate:
                waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)
            if waveform.shape[0] == 1:
                waveform = waveform.repeat(2, 1)
            elif waveform.shape[0] > 2:
                waveform = waveform[:2, :]
            if waveform.shape[-1] < self.segment_samples:
                waveform = F.pad(waveform, (0, self.segment_samples - waveform.shape[-1]))
            return waveform[:, :self.segment_samples]
        except Exception:
            # Fallback en cas d'erreur de lecture
            return torch.zeros(2, self.segment_samples)

    def _compute_stft_magnitude(self, waveform: torch.Tensor):
        window = self.window.to(waveform.device)
        stft = torch.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=window,
            return_complex=True
        )
        mag = torch.abs(stft)
        # 1024 premiers bacs de fréquence (puissance de 2)
        return mag[:, :1024, :]

    def __getitem__(self, idx):
        if len(self.vocal_files) == 0 or len(self.instrumental_files) == 0:
            # Mode synthétique test
            t_frames = 512
            mix_mag = torch.rand(2, 1024, t_frames) * 2.0
            voc_mag = mix_mag * torch.rand(2, 1024, t_frames)
            ins_mag = torch.clamp(mix_mag - voc_mag, min=0.0)
            return mix_mag, voc_mag, ins_mag

        # Sélection aléatoire pour remix dynamique
        voc_path = random.choice(self.vocal_files) if self.augment else self.vocal_files[idx % len(self.vocal_files)]
        ins_path = random.choice(self.instrumental_files) if self.augment else self.instrumental_files[idx % len(self.instrumental_files)]

        voc_audio = self._load_audio_chunk(voc_path)
        ins_audio = self._load_audio_chunk(ins_path)

        # Augmentation : Gain aléatoire indépendant
        if self.augment:
            gain_voc = random.uniform(0.7, 1.3)
            gain_ins = random.uniform(0.7, 1.3)
            voc_audio = voc_audio * gain_voc
            ins_audio = ins_audio * gain_ins

            # Inversion gauche/droite aléatoire
            if random.random() < 0.5:
                voc_audio = torch.flip(voc_audio, dims=[0])
                ins_audio = torch.flip(ins_audio, dims=[0])

        mix_audio = voc_audio + ins_audio

        # Normalisation transparente pour éviter tout clipping
        max_peak = torch.max(torch.abs(mix_audio)).item()
        if max_peak > 0.95:
            scale = 0.95 / max_peak
            mix_audio = mix_audio * scale
            voc_audio = voc_audio * scale
            ins_audio = ins_audio * scale

        mix_mag = self._compute_stft_magnitude(mix_audio)
        voc_mag = self._compute_stft_magnitude(voc_audio)
        ins_mag = self._compute_stft_magnitude(ins_audio)

        return mix_mag, voc_mag, ins_mag
