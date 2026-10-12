// public/sw.js - Service Worker de Voksplit pour le mode PWA et cache offline

const CACHE_NAME = 'voksplit-shell-v2';
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
  // Ignorer les requêtes non-HTTP(S) (ex: extensions Chrome chrome-extension://, firefox moz-extension://)
  if (!e.request.url.startsWith('http://') && !e.request.url.startsWith('https://')) {
    return;
  }

  if (e.request.method !== 'GET') {
    return;
  }

  const url = new URL(e.request.url);

  // Ignorer les requêtes externes et modèles lourds pour ce cache shell
  if (url.origin !== self.location.origin || url.hostname.includes('huggingface.co') || url.pathname.includes('/ort-wasm/')) {
    return;
  }

  e.respondWith(
    caches.match(e.request).then((res) => {
      return (
        res ||
        fetch(e.request).then((netRes) => {
          if (netRes && netRes.status === 200) {
            const clone = netRes.clone();
            caches.open(CACHE_NAME).then((c) => c.put(e.request, clone).catch(() => {})).catch(() => {});
          }
          return netRes;
        })
      );
    }).catch(() => caches.match('/'))
  );
});
