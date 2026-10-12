# tools/training/export_onnx.py
# Script d'export, d'optimisation FP16 et de calcul SHA-256 pour VoksplitNet

import os
import hashlib
import time
import argparse
import torch
from model import VoksplitNet

def compute_sha256(filepath):
    sha = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            sha.update(chunk)
    return sha.hexdigest()

def export_model(checkpoint_path=None, output_dir="./exported_models"):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cpu")

    print("=== Export de VoksplitNet vers ONNX & FP16 WebGPU ===")
    model = VoksplitNet(in_channels=2, base_channels=32, num_freq_bins=1024)

    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Chargement des poids entraînés : {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=device)
        state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        model.load_state_dict(state_dict)
    else:
        print("Avertissement : Aucun checkpoint fourni ou introuvable. Export avec poids initialisés.")

    model.eval()

    # Tenseur factice d'entrée : [Batch=1, Channels=2 (Stéréo), Freq=1024, Time=512 (~6 secondes)]
    dummy_input = torch.randn(1, 2, 1024, 512, device=device)

    fp32_onnx_path = os.path.join(output_dir, "voksplit-net.onnx")
    fp16_onnx_path = os.path.join(output_dir, "voksplit-net-fp16.onnx")

    print(f"\n1. Export ONNX standard (FP32) vers : {fp32_onnx_path}")
    torch.onnx.export(
        model,
        dummy_input,
        fp32_onnx_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["mix_mag"],
        output_names=["mask_vocals", "mask_instrumental"],
        dynamic_axes={
            "mix_mag": {0: "batch_size", 3: "time_frames"},
            "mask_vocals": {0: "batch_size", 3: "time_frames"},
            "mask_instrumental": {0: "batch_size", 3: "time_frames"}
        }
    )

    size_fp32_mb = os.path.getsize(fp32_onnx_path) / (1024 * 1024)
    print(f"  -> Taille FP32 : {size_fp32_mb:.2f} Mo")

    # Conversion en FP16 pour réduire de 50% la taille et accélérer le WebGPU
    print(f"\n2. Optimisation FP16 pour WebGPU...")
    try:
        import onnx
        from onnxconverter_common import float16
        onnx_model = onnx.load(fp32_onnx_path)
        onnx_model_fp16 = float16.convert_float_to_float16(onnx_model, keep_io_types=True)
        onnx.save(onnx_model_fp16, fp16_onnx_path)
        size_fp16_mb = os.path.getsize(fp16_onnx_path) / (1024 * 1024)
        print(f"  -> Taille FP16 optimisée : {size_fp16_mb:.2f} Mo (Gain de {((size_fp32_mb - size_fp16_mb) / size_fp32_mb) * 100:.1f}%)")
    except ImportError:
        print("  -> onnxconverter-common non installé. Utilisation de la version standard pour les tests.")
        fp16_onnx_path = fp32_onnx_path

    # Benchmark d'inférence avec onnxruntime
    print(f"\n3. Benchmark d'inférence de vérification...")
    try:
        import onnxruntime as ort
        import numpy as np

        session = ort.InferenceSession(fp16_onnx_path, providers=["CPUExecutionProvider"])
        test_in = np.random.randn(1, 2, 1024, 512).astype(np.float32)

        # Warmup
        _ = session.run(None, {"mix_mag": test_in})

        # Mesure temps d'exécution
        t0 = time.time()
        for _ in range(5):
            out_voc, out_ins = session.run(None, {"mix_mag": test_in})
        avg_time_ms = ((time.time() - t0) / 5) * 1000
        print(f"  -> Temps d'inférence CPU moyen pour 6s d'audio : {avg_time_ms:.1f} ms (~{avg_time_ms/6000:.2f}x temps réel)")
        print(f"  -> Sortie Vocaux shape : {out_voc.shape}, valeurs min={out_voc.min():.3f}, max={out_voc.max():.3f}")
        print(f"  -> Sortie Instru shape : {out_ins.shape}, valeurs min={out_ins.min():.3f}, max={out_ins.max():.3f}")
    except ImportError:
        print("  -> onnxruntime non disponible localement pour le benchmark (peut être exécuté sous Colab).")

    # Calcul SHA-256
    print(f"\n4. Calcul des signatures d'intégrité SHA-256 :")
    hash_fp32 = compute_sha256(fp32_onnx_path)
    print(f"  [voksplit-net.onnx]        SHA-256 : {hash_fp32}")
    if os.path.exists(fp16_onnx_path) and fp16_onnx_path != fp32_onnx_path:
        hash_fp16 = compute_sha256(fp16_onnx_path)
        print(f"  [voksplit-net-fp16.onnx]   SHA-256 : {hash_fp16}")

    print("\n✅ Export terminé avec succès !")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export ONNX pour VoksplitNet")
    parser.add_argument("--checkpoint", type=str, default="./checkpoints/best_voksplit_model.pth", help="Chemin du checkpoint PyTorch")
    parser.add_argument("--output_dir", type=str, default="./exported_models", help="Dossier de sortie")
    args = parser.parse_args()
    export_model(args.checkpoint, args.output_dir)
