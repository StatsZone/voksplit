# 🎛️ VoksplitNet : Architecture, Entraînement & Export IA

Ce dossier contient l'ensemble du pipeline de deep learning pour entraîner et exporter le modèle propriétaire **VoksplitNet**, conçu pour la séparation stéréo de haute précision **Voix vs Instrumental** directement dans le navigateur.

---

## 🔬 1. Architecture du Modèle : TFC-TDF U-Net

`VoksplitNet` est une architecture hybride optimisée pour les contraintes de mémoire et de vitesse du Web :
- **TFC (Time-Frequency Convolutions)** : Convolutions résiduelles 2D capturant la dynamique locale temps-fréquence.
- **TDF (Time-Distributed Fully Connected)** : Couches linéaires sur l'axe des fréquences pour capter les relations harmoniques globales (octaves, harmoniques vocales, formants) à travers le spectre audio.
- **Taille & Poids** :
  - **~3.2 à 8M paramètres**
  - **~13 Mo en FP32** / **~6.5 à 18 Mo en FP16 WebGPU**
  - **100% Inférence Client-Side** : Zéro octet audio envoyé sur un serveur.

---

## ⚖️ 2. Conformité Légale & Dataset MUSDB18-HQ

Contrairement aux modèles communautaires aux licences ambiguës ou restrictives, VoksplitNet est entraîné sur le dataset certifié **MUSDB18-HQ** :
- **150 morceaux complets** non compressés (44.1 kHz, stéréo, format WAV 16-bit).
- Pistes isolées : `vocals.wav`, `drums.wav`, `bass.wav`, `other.wav`.
- Augmentation de données par **Remix Augmentation** : mixage dynamique d'une voix issue d'un morceau avec un instrumental issu d'un autre morceau, créant plus de 10 000 combinaisons acoustiques virtuelles.

Pour télécharger le dataset MUSDB18-HQ :
👉 [https://zenodo.org/record/3338373](https://zenodo.org/record/3338373) (~30 Go décompressé)

---

## 🚀 3. Méthode Recommandée : Entraînement 1-Clic sur Google Colab

Aucun GPU surpuissant n'est nécessaire sur votre machine locale ! Vous pouvez utiliser un GPU Tesla T4 gratuit sur Google Colab :

1. Ouvrez [Google Colab](https://colab.research.google.com/).
2. Chargez le fichier `tools/training/Voksplit_Training_Colab.ipynb`.
3. Allez dans **Exécution** > **Modifier le type d'exécution** > Choisissez **T4 GPU**.
4. Cliquez sur **Exécuter tout**.
5. À la fin de l'exécution, le fichier optimisé `voksplit-net-fp16.onnx` sera automatiquement téléchargé sur votre ordinateur avec sa clé SHA-256.

---

## 💻 4. Entraînement Local (Optionnel)

Si vous disposez d'un GPU Nvidia local avec CUDA :

```bash
# 1. Installation des dépendances
pip install torch torchaudio onnx onnxruntime onnxconverter-common

# 2. Lancer l'entraînement sur MUSDB18-HQ
python train.py --data_dir ./data/musdb18hq --epochs 50 --batch_size 8 --lr 0.001

# 3. Exporter le meilleur checkpoint vers ONNX & FP16
python export_onnx.py --checkpoint ./checkpoints/best_voksplit_model.pth --output_dir ./exported_models
```

---

## 🌐 5. Intégration dans l'Application Voksplit

Une fois `voksplit-net-fp16.onnx` généré :
1. Placez le fichier dans `public/models/voksplit-net-fp16.onnx` ou hébergez-le sur votre bucket CDN / GitHub Releases.
2. Notez la signature SHA-256 générée par `export_onnx.py`.
3. Mettez à jour `public/stem-worker.js` avec l'URL et la signature SHA-256 :
```javascript
const MODEL_URL = '/models/voksplit-net-fp16.onnx';
const MODEL_SHA256 = '<VOTRE_HASH_SHA256>';
```
Le Web Worker vérifiera l'intégrité cryptographique avant d'exécuter l'isolation en WebGPU avec fallback multithread WASM SIMD.
