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

## ⚖️ 2. Conformité Légale & Approche Commerciale (100% Libre de Droits)

Contrairement aux modèles de recherche limités par des licences non-commerciales restrictives (ex: CC-BY-NC-SA de MUSDB18 ou CC-BY-NC 4.0 de Meta), VoksplitNet repose sur une approche **100% Commercial-Compliant** basée sur le **Dynamic Synthetic Remixing** :

- **Pistes Vocales Libres (CC0 / CC-BY / Domaine Public)** : Acapellas issues de plateformes ouvertes (ccMixter, VocalSet, Mozilla Common Voice).
- **Pistes Instrumentales Libres (CC0 / CC-BY)** : Morceaux instrumentaux et stems de Slakh2100, Free Music Archive (FMA CC-BY/CC0).
- **Dynamic Synthetic Remixing ($M = V + I$)** :
  - Chaque batch d'entraînement associe aléatoirement une voix avec une piste instrumentale indépendante.
  - Génère une combinatoire quasi-infinie (ex: 500 voix × 500 instrumentaux = **250 000 morceaux virtuels uniques**).
  - Variations dynamiques de gain (0.7x à 1.3x), d'équilibrage et de panoramique stéréo.
- **Zéro risque juridique** : Aucune donnée sous clause "NC" (Non-Commercial) n'est injectée dans les poids du modèle.

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
pip install torch torchaudio onnx onnxruntime onnxconverter-common onnxscript

# 2. Lancer l'entraînement avec le dataset commercial
python train.py --data_dir ./data/commercial_stems --epochs 50 --batch_size 8 --lr 0.001

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
