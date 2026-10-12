# 🎧 Voksplit — by Sound Capsule

Application hybride IA de séparation audio (voix & instruments) en temps réel, 100 % gratuite, sans cloud et respectueuse de la vie privée.

---

## ⚡ Caractéristiques
- **100 % Local & Gratuit :** Aucune donnée audio n'est envoyée sur un serveur tiers. L'IA s'exécute directement sur votre processeur / GPU via WebGPU et WebAssembly.
- **Temps Réel & Haute Fidélité :** Double moteur hybride (IA HTDemucs et filtre instantané DSP).
- **Lecteur A/B Synchronisé :** Bascule instantanée sans clic ni latence entre l'original, l'instrumental et la voix isolée.
- **PWA & Offline :** Installable sur mobile et ordinateur, accessible partout via Cloudflare.

---

## 🛠️ Stack Technique
- **Framework :** Next.js 15 (App Router, TypeScript strict, Tailwind CSS)
- **Moteur IA :** ONNX Runtime Web, Demucs Web Worker, WebGPU / WASM SIMD
- **Hébergement :** Cloudflare Workers & Pages
- **Stockage Local :** IndexedDB / OPFS

---

## 💻 Développement Local
```bash
npm install
npm run dev
```

Déploiement statique :
```bash
npm run build
```
