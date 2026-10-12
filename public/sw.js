// public/sw.js - Service Worker de Voksplit pour le mode PWA et cache offline

const CACHE_NAME = 'voksplit-shell-v1';
const STATIC_ASSETS = [
  '/',
  '/manifest.json',
  '/stem-worker.js',
  '/vendor/demucs-web/index.js',
  '/vendor/demucs-web/constants.js',
  '/vendor/demucs-web/fft.js',
  '/vendor/demucs-web/processor.js'
];

self.addEventListener('install', (e) => {
  e.waitUntil(
    caches.open(CACHE_NAME).then((cache) => cache.addAll(STATIC_ASSETS)).then(() => self.skipWaiting())
  );
});

self.addEventListener('activate', (e) => {
  e.waitUntil(
    caches.keys().then((keys) =>
      Promise.all(keys.map((k) => (k !== CACHE_NAME && !k.startsWith('voksplit-ai-model') ? caches.delete(k) : null)))
    ).then(() => self.clients.claim())
  );
});

self.addEventListener('fetch', (e) => {
  const url = new URL(e.request.url);

  // Le modèle IA et les binaires lourds sont gérés par leur propre système de cache immuable
  if (url.hostname.includes('huggingface.co') || url.pathname.includes('/ort-wasm/')) {
    return;
  }

  e.respondWith(
    caches.match(e.request).then((res) => {
      return res || fetch(e.request).then((netRes) => {
        if (netRes && netRes.status === 200 && e.request.method === 'GET') {
          const clone = netRes.clone();
          caches.open(CACHE_NAME).then((c) => c.put(e.request, clone));
        }
        return netRes;
      });
    }).catch(() => caches.match('/'))
  );
});
