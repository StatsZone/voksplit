# tools/training/dataset.py
# Chargement du dataset MUSDB18-HQ avec STFT, augmentation de données et chunking audio

import os
import glob
import random
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchaudio

class MUSDB18Dataset(Dataset):
    """
    Dataset PyTorch pour MUSDB18-HQ.
    Structure attendue :
      root_dir/
        train/ (ou test/)
          track_name_1/
            vocals.wav
            drums.wav
            bass.wav
            other.wav
            mixture.wav
    """
    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        sample_rate: int = 44100,
        segment_seconds: float = 6.0,
        n_fft: int = 2048,
        hop_length: int = 512,
        augment: bool = True,
        remix_augmentation: bool = True
    ):
        super().__init__()
        self.root_dir = root_dir
        self.split = split
        self.sample_rate = sample_rate
        self.segment_samples = int(segment_seconds * sample_rate)
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.augment = augment and (split == "train")
        self.remix_augmentation = remix_augmentation and (split == "train")
        self.window = torch.hann_window(n_fft)

        split_dir = os.path.join(root_dir, split)
        if os.path.exists(split_dir):
            self.track_dirs = sorted([
                d for d in glob.glob(os.path.join(split_dir, "*"))
                if os.path.isdir(d) and os.path.exists(os.path.join(d, "vocals.wav"))
            ])
        else:
            self.track_dirs = []

        print(f"[{split.upper()}] {len(self.track_dirs)} morceaux MUSDB18-HQ détectés dans {split_dir}")

    def __len__(self):
        if len(self.track_dirs) == 0:
            return 100 # Mode démo / synthétique si pas de dataset
        # Multiplie virtuellement les itérations par époque grâce au découpage aléatoire
        return len(self.track_dirs) * (10 if self.split == "train" else 1)

    def _load_audio_chunk(self, path: str, start_sample: int, num_samples: int):
        info = torchaudio.info(path)
        total_samples = info.num_frames
        
        if total_samples <= num_samples:
            waveform, sr = torchaudio.load(path)
            if sr != self.sample_rate:
                waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)
            # Pad si trop court
            if waveform.shape[-1] < num_samples:
                waveform = F.pad(waveform, (0, num_samples - waveform.shape[-1]))
            else:
                waveform = waveform[:, :num_samples]
            return waveform

        # Découpe ciblée
        waveform, sr = torchaudio.load(path, frame_offset=start_sample, num_frames=num_samples)
        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)
        if waveform.shape[-1] < num_samples:
            waveform = F.pad(waveform, (0, num_samples - waveform.shape[-1]))
        return waveform[:, :num_samples]

    def _compute_stft_magnitude(self, waveform: torch.Tensor):
        # waveform: [2, Samples]
        # STFT complexe: [2, F=1025, T]
        window = self.window.to(waveform.device)
        stft = torch.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=window,
            return_complex=True
        )
        # Magnitude
        mag = torch.abs(stft) # [2, 1025, T]
        # On ne conserve que les 1024 premiers bacs de fréquence (puissance de 2 pour U-Net)
        return mag[:, :1024, :]

    def __getitem__(self, idx):
        if len(self.track_dirs) == 0:
            # Synthétique pour tests rapides sans dataset
            t_frames = 512
            mix_mag = torch.rand(2, 1024, t_frames) * 2.0
            voc_mag = mix_mag * torch.rand(2, 1024, t_frames)
            ins_mag = torch.clamp(mix_mag - voc_mag, min=0.0)
            return mix_mag, voc_mag, ins_mag

        track_idx = idx % len(self.track_dirs)
        track_dir = self.track_dirs[track_idx]

        # Détermination de l'offset temporel aléatoire
        voc_path = os.path.join(track_dir, "vocals.wav")
        info = torchaudio.info(voc_path)
        max_offset = max(0, info.num_frames - self.segment_samples)
        offset = random.randint(0, max_offset) if (self.augment and max_offset > 0) else 0

        # Chargement Voix
        voc_audio = self._load_audio_chunk(voc_path, offset, self.segment_samples)

        # Remix Augmentation : possibilité de mixer la voix avec l'instrumental d'un autre morceau
        if self.remix_augmentation and random.random() < 0.5:
            ins_track_idx = random.randint(0, len(self.track_dirs) - 1)
            ins_dir = self.track_dirs[ins_track_idx]
            ins_info = torchaudio.info(os.path.join(ins_dir, "drums.wav"))
            ins_offset = random.randint(0, max(0, ins_info.num_frames - self.segment_samples))
        else:
            ins_dir = track_dir
            ins_offset = offset

        # Instrumental = Drums + Bass + Other
        drums = self._load_audio_chunk(os.path.join(ins_dir, "drums.wav"), ins_offset, self.segment_samples)
        bass = self._load_audio_chunk(os.path.join(ins_dir, "bass.wav"), ins_offset, self.segment_samples)
        other = self._load_audio_chunk(os.path.join(ins_dir, "other.wav"), ins_offset, self.segment_samples)
        ins_audio = drums + bass + other

        # Augmentation : Gain aléatoire sur chaque source
        if self.augment:
            gain_voc = random.uniform(0.7, 1.3)
            gain_ins = random.uniform(0.7, 1.3)
            voc_audio = voc_audio * gain_voc
            ins_audio = ins_audio * gain_ins

            # Inversion gauche/droite aléatoire (panning)
            if random.random() < 0.5:
                voc_audio = torch.flip(voc_audio, dims=[0])
                ins_audio = torch.flip(ins_audio, dims=[0])

        mix_audio = voc_audio + ins_audio

        # Calcul des spectrogrammes de magnitude
        mix_mag = self._compute_stft_magnitude(mix_audio)
        voc_mag = self._compute_stft_magnitude(voc_audio)
        ins_mag = self._compute_stft_magnitude(ins_audio)

        return mix_mag, voc_mag, ins_mag
