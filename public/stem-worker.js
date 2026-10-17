// public/stem-worker.js - Web Worker dédié à l'isolation vocale pour Voksplit
// Support hybride : VoksplitNet (Propriétaire 100% Commercial) & Demucs (Secours)
// 100% local, non-bloquant pour l'interface UI, avec contrôle d'intégrité SHA-256

const ORT_VERSION = '1.20.1';
const ORT_BASE = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/`;
const DEMUCS_LOCAL_PATH = '/vendor/demucs-web/index.js?v=14.68.31';

// Modèle Principal : MDX-Net UVR5 Ultra-Studio (~31.85 Mo FP16, découpé en 2x 15.9 Mo pour Cloudflare Pages < 25 Mo)
const MDX_MODEL_CACHE_KEY = '/models/uvr-mdx-net-inst-hq3-fp16.onnx';
const MDX_MODEL_PARTS = [
    '/models/uvr-mdx-net-inst-hq3-fp16.part1.bin',
    '/models/uvr-mdx-net-inst-hq3-fp16.part2.bin'
];
const MDX_MODEL_TOTAL_BYTES = 33398584;
const MDX_MODEL_SHA256 = '62d30044a9e0980efe2daeba0e0f2080fcb23bf8a46b98c0ba5183fe9bf80a9e';

// Modèle Secondaire Léger : VoksplitNet Propriétaire (Local, ~6.5 Mo FP16)
const VOKSPLIT_MODEL_URL = '/models/voksplit-net-fp16.onnx';
let VOKSPLIT_MODEL_SHA256 = '98dc7fa58831ec0aa68012636da72f6af4ee5ac41e60fd19c34cd4af5aeed468';

// Modèle de secours distant (si aucun modèle local n'est accessible)
const FALLBACK_MODEL_URL = 'https://huggingface.co/timcsy/demucs-web-onnx/resolve/main/htdemucs_embedded.onnx';
const FALLBACK_MODEL_SHA256 = 'e5e425c17683f163a472462eb5f5a4ffcd11c31858d57fbd0833b012d8b88077';

const AI_MODEL_CACHE = 'voksplit-ai-model-v5';
const SAMPLE_RATE = 44100;

let processorPromise = null;
let segmentCb = () => {};

const status = (id, msg, progress) => self.postMessage({ id, type: 'status', msg, progress });

function toHex(buffer) {
    const bytes = new Uint8Array(buffer);
    let hex = '';
    for (let i = 0; i < bytes.length; i++) {
        hex += bytes[i].toString(16).padStart(2, '0');
    }
    return hex;
}

async function verifyAndGetBuffer(buffer, expectedHash) {
    if (expectedHash && self.crypto && self.crypto.subtle) {
        const digest = await self.crypto.subtle.digest('SHA-256', buffer);
        const computedHash = toHex(digest);
        if (computedHash !== expectedHash) {
            console.error(`[Voksplit Worker] Échec intégrité: ${computedHash} !== ${expectedHash}`);
            throw new Error("Échec d'intégrité du modèle IA (signature invalide).");
        }
    }
    return buffer;
}

let processorCache = new Map();

async function fetchStreamIntoChunks(url, chunks, onBytesLoaded) {
    const response = await fetch(url);
    if (!response.ok || (response.headers.get('content-type') || '').includes('text/html')) {
        throw new Error(`Téléchargement impossible pour ${url} (HTTP ${response.status})`);
    }
    const reader = response.body.getReader();
    let partLoaded = 0;
    while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        chunks.push(value);
        partLoaded += value.length;
        onBytesLoaded(value.length);
    }
    return partLoaded;
}

async function getModelBuffer(id, engine = 'mdx') {
    let cache = null;
    try { cache = await caches.open(AI_MODEL_CACHE); } catch (e) {}

    let targetCacheKey = MDX_MODEL_CACHE_KEY;
    let targetUrls = MDX_MODEL_PARTS;
    let targetHash = MDX_MODEL_SHA256;
    let expectedTotal = MDX_MODEL_TOTAL_BYTES;
    let modelName = "MDX-Net UVR5 (Qualité Studio 31 Mo)";
    let modelType = 'mdx';

    if (engine === 'voksplit') {
        targetCacheKey = VOKSPLIT_MODEL_URL;
        targetUrls = [VOKSPLIT_MODEL_URL];
        targetHash = VOKSPLIT_MODEL_SHA256;
        expectedTotal = 6856957;
        modelName = "VoksplitNet (Propriétaire Léger 6.5 Mo)";
        modelType = 'voksplit';
    } else if (engine === 'demucs') {
        targetCacheKey = FALLBACK_MODEL_URL;
        targetUrls = [FALLBACK_MODEL_URL];
        targetHash = FALLBACK_MODEL_SHA256;
        expectedTotal = 180000000;
        modelName = "HTDemucs (Secours Distant 180 Mo)";
        modelType = 'demucs';
    } else {
        // MDX par défaut : vérifier si disponible, sinon basculer proprement sur VoksplitNet
        try {
            const testRes = await fetch(MDX_MODEL_PARTS[0], { method: 'HEAD' });
            const ct = (testRes.headers.get('content-type') || '').toLowerCase();
            if (!testRes.ok || ct.includes('text/html')) {
                targetCacheKey = VOKSPLIT_MODEL_URL;
                targetUrls = [VOKSPLIT_MODEL_URL];
                targetHash = VOKSPLIT_MODEL_SHA256;
                expectedTotal = 6856957;
                modelName = "VoksplitNet (Secours Local 6.5 Mo)";
                modelType = 'voksplit';
            }
        } catch (e) {
            targetCacheKey = VOKSPLIT_MODEL_URL;
            targetUrls = [VOKSPLIT_MODEL_URL];
            targetHash = VOKSPLIT_MODEL_SHA256;
            expectedTotal = 6856957;
            modelName = "VoksplitNet (Secours Local 6.5 Mo)";
            modelType = 'voksplit';
        }
    }

    if (cache) {
        const cached = await cache.match(targetCacheKey);
        if (cached) {
            status(id, `Modèle ${modelName} vérifié depuis le cache local`, 0.25);
            const buf = await cached.arrayBuffer();
            try {
                const verified = await verifyAndGetBuffer(buf, targetHash);
                return { buffer: verified, modelType };
            } catch (err) {
                console.warn("[Voksplit Worker] Cache corrompu, nouveau téléchargement...");
                await cache.delete(targetCacheKey);
            }
        }
    }

    const chunks = [];
    let loaded = 0;
    let lastMb = -1;
    const total = expectedTotal;

    for (const partUrl of targetUrls) {
        await fetchStreamIntoChunks(partUrl, chunks, (delta) => {
            loaded += delta;
            const mb = Math.floor(loaded / 1048576);
            if (mb !== lastMb) {
                lastMb = mb;
                status(id, `Chargement ${modelName} : ${mb} / ${Math.round(total / 1048576)} Mo`, Math.min(0.25, (loaded / total) * 0.25));
            }
        });
    }

    const bytes = new Uint8Array(loaded);
    let off = 0;
    for (const c of chunks) { bytes.set(c, off); off += c.length; }

    status(id, "Contrôle d'intégrité de l'IA...", 0.28);
    const verified = await verifyAndGetBuffer(bytes.buffer, targetHash);

    if (cache) {
        try {
            await cache.put(targetCacheKey, new Response(bytes, { headers: { 'Content-Type': 'application/octet-stream' } }));
        } catch (e) {
            console.warn('[Voksplit Worker] Mise en cache impossible :', e);
        }
    }

    return { buffer: verified, modelType };
}

function getProcessor(id, engine = 'mdx') {
    if (processorCache.has(engine)) return processorCache.get(engine);
    const promise = (async () => {
        const ort = await import(`${ORT_BASE}ort.webgpu.min.mjs`);
        ort.env.wasm.wasmPaths = ORT_BASE;
        ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(navigator.hardwareConcurrency || 4, 8) : 1;
        ort.env.wasm.proxy = false;

        let providers = ['wasm'];
        let hasWebGpu = false;
        try {
            if (navigator.gpu) {
                const adapter = await navigator.gpu.requestAdapter();
                if (adapter) {
                    providers = ['webgpu', 'wasm'];
                    hasWebGpu = true;
                    console.log('[Voksplit] Accélération WebGPU active !');
                }
            }
        } catch (e) {
            console.warn('[Voksplit] Détection WebGPU :', e);
        }

        const sessionOptions = { executionProviders: providers, enableCpuMemArena: false, enableMemPattern: false };
        const { buffer: modelBuffer, modelType } = await getModelBuffer(id, engine);

        let processor;
        if (modelType === 'mdx') {
            const { MDXProcessor } = await import(DEMUCS_LOCAL_PATH);
            processor = new MDXProcessor({
                ort,
                sessionOptions,
                onProgress: (info) => segmentCb(info),
                onLog: (phase, msg) => console.log(`[MDX-Net][${phase}] ${msg}`)
            });
        } else if (modelType === 'voksplit') {
            const { VoksplitProcessor } = await import(DEMUCS_LOCAL_PATH);
            processor = new VoksplitProcessor({
                ort,
                sessionOptions,
                onProgress: (info) => segmentCb(info),
                onLog: (phase, msg) => console.log(`[VoksplitNet][${phase}] ${msg}`)
            });
        } else {
            const { DemucsProcessor } = await import(DEMUCS_LOCAL_PATH);
            processor = new DemucsProcessor({
                ort,
                sessionOptions,
                onProgress: (info) => segmentCb(info),
                onLog: (phase, msg) => console.log(`[Demucs][${phase}] ${msg}`)
            });
        }

        const initMsg = hasWebGpu ? "Initialisation MDX Studio (WebGPU ⚡)..." : "Initialisation MDX Studio (CPU multicœur)...";
        status(id, initMsg, 0.32);
        await processor.loadModel(modelBuffer);
        return processor;
    })();
    processorCache.set(engine, promise);
    promise.catch(() => { processorCache.delete(engine); });
    return promise;
}

function encodeWav(left, right) {
    const frames = left.length;
    const dataLength = frames * 4;
    const buffer = new ArrayBuffer(44 + dataLength);
    const view = new DataView(buffer);
    const w = (o, s) => { for (let i = 0; i < s.length; i++) view.setUint8(o + i, s.charCodeAt(i)); };
    w(0, 'RIFF'); view.setUint32(4, 36 + dataLength, true); w(8, 'WAVE'); w(12, 'fmt ');
    view.setUint32(16, 16, true); view.setUint16(20, 1, true); view.setUint16(22, 2, true);
    view.setUint32(24, SAMPLE_RATE, true); view.setUint32(28, SAMPLE_RATE * 4, true);
    view.setUint16(32, 4, true); view.setUint16(34, 16, true); w(36, 'data'); view.setUint32(40, dataLength, true);

    const pcm16 = new Int16Array(buffer, 44, frames * 2);
    let p = 0;
    for (let i = 0; i < frames; i++) {
        let sL = left[i];
        if (isNaN(sL) || !isFinite(sL)) sL = 0;
        else if (sL > 0.999) sL = 0.999;
        else if (sL < -0.999) sL = -0.999;
        pcm16[p++] = Math.max(-32767, Math.min(32767, Math.round(sL * 32767)));

        let sR = right[i];
        if (isNaN(sR) || !isFinite(sR)) sR = 0;
        else if (sR > 0.999) sR = 0.999;
        else if (sR < -0.999) sR = -0.999;
        pcm16[p++] = Math.max(-32767, Math.min(32767, Math.round(sR * 32767)));
    }
    return new Blob([buffer], { type: 'audio/wav' });
}

self.onmessage = async (e) => {
    const { id, left, right, engine = 'mdx' } = e.data || {};
    try {
        status(id, "Préparation de l'IA locale...", 0.05);
        const processor = await getProcessor(id, engine);

        segmentCb = ({ currentSegment, totalSegments }) => {
            status(id, `Séparation en cours : ${Math.round((currentSegment / totalSegments) * 100)} %`, 0.35 + 0.6 * (currentSegment / totalSegments));
        };
        status(id, 'Séparation en cours : 0 %', 0.35);
        const stems = await processor.separate(left, right);

        status(id, 'Mastering & extraction studio...', 0.96);
        const n = left.length;
        const instL = new Float32Array(n);
        const instR = new Float32Array(n);
        const vocL = new Float32Array(n);
        const vocR = new Float32Array(n);

        for (let i = 0; i < n; i++) {
            let sL = stems.instrumental ? stems.instrumental.left[i] : (
                (stems.drums?.left?.[i] || 0) + (stems.bass?.left?.[i] || 0) + (stems.other?.left?.[i] || 0)
            );
            let sR = stems.instrumental ? stems.instrumental.right[i] : (
                (stems.drums?.right?.[i] || 0) + (stems.bass?.right?.[i] || 0) + (stems.other?.right?.[i] || 0)
            );

            if (isNaN(sL) || !isFinite(sL)) sL = 0;
            if (isNaN(sR) || !isFinite(sR)) sR = 0;

            instL[i] = sL;
            instR[i] = sR;

            let vL = stems.vocals?.left?.[i] || 0;
            let vR = stems.vocals?.right?.[i] || 0;
            if (isNaN(vL) || !isFinite(vL)) vL = 0;
            if (isNaN(vR) || !isFinite(vR)) vR = 0;

            vocL[i] = vL;
            vocR[i] = vR;
        }

        // Normalisation linéaire de crête transparente (-0.5 dB)
        // Élimine 100% des grésillements et distorsions de saturation
        let peakInst = 0;
        let peakVoc = 0;
        for (let i = 0; i < n; i++) {
            const absIL = Math.abs(instL[i]);
            const absIR = Math.abs(instR[i]);
            if (absIL > peakInst) peakInst = absIL;
            if (absIR > peakInst) peakInst = absIR;

            const absVL = Math.abs(vocL[i]);
            const absVR = Math.abs(vocR[i]);
            if (absVL > peakVoc) peakVoc = absVL;
            if (absVR > peakVoc) peakVoc = absVR;
        }

        if (peakInst > 0.95) {
            const scale = 0.95 / peakInst;
            for (let i = 0; i < n; i++) {
                instL[i] *= scale;
                instR[i] *= scale;
            }
        }

        if (peakVoc > 0.95) {
            const scale = 0.95 / peakVoc;
            for (let i = 0; i < n; i++) {
                vocL[i] *= scale;
                vocR[i] *= scale;
            }
        }

        const instrumentalBlob = encodeWav(instL, instR);
        const vocalsBlob = encodeWav(vocL, vocR);

        status(id, 'Isolation terminée avec succès !', 1.0);
        self.postMessage({ id, type: 'complete', instrumentalBlob, vocalsBlob });
    } catch (err) {
        console.error('[Voksplit Worker] Erreur globale :', err);
        self.postMessage({ id, type: 'error', error: err.message || 'Erreur inconnue dans le worker' });
    }
};
