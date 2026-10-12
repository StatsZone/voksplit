// public/stem-worker.js - Web Worker dédié à l'isolation vocale pour Voksplit
// 100% local, non-bloquant pour l'interface UI, avec contrôle d'intégrité SHA-256

const ORT_VERSION = '1.20.1';
const ORT_BASE = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/`;
const DEMUCS_LOCAL_PATH = '/vendor/demucs-web/index.js';
const MODEL_URL = 'https://huggingface.co/timcsy/demucs-web-onnx/resolve/main/htdemucs_embedded.onnx';
const MODEL_SHA256 = 'e5e425c17683f163a472462eb5f5a4ffcd11c31858d57fbd0833b012d8b88077';
const AI_MODEL_CACHE = 'voksplit-ai-model-v1';
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

async function verifyAndGetBuffer(buffer) {
    if (self.crypto && self.crypto.subtle) {
        const digest = await self.crypto.subtle.digest('SHA-256', buffer);
        const computedHash = toHex(digest);
        if (computedHash !== MODEL_SHA256) {
            console.error(`[Voksplit Worker] Échec intégrité: ${computedHash} !== ${MODEL_SHA256}`);
            throw new Error("Échec d'intégrité du modèle IA (signature invalide).");
        }
    }
    return buffer;
}

async function getModelBuffer(id) {
    let cache = null;
    try { cache = await caches.open(AI_MODEL_CACHE); } catch (e) {}
    if (cache) {
        const cached = await cache.match(MODEL_URL);
        if (cached) {
            status(id, 'Modèle IA vérifié depuis le cache local', 0.25);
            const buf = await cached.arrayBuffer();
            try {
                return await verifyAndGetBuffer(buf);
            } catch (err) {
                console.warn("[Voksplit Worker] Cache corrompu, nouveau téléchargement...");
                await cache.delete(MODEL_URL);
            }
        }
    }

    const response = await fetch(MODEL_URL);
    if (!response.ok) throw new Error(`Téléchargement du modèle IA impossible (HTTP ${response.status})`);
    const total = parseInt(response.headers.get('Content-Length') || '0', 10) || 180534758;
    const reader = response.body.getReader();
    const chunks = [];
    let loaded = 0;
    let lastMb = -1;
    while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        chunks.push(value);
        loaded += value.length;
        const mb = Math.floor(loaded / 1048576);
        if (mb !== lastMb) {
            lastMb = mb;
            status(id, `Téléchargement IA (1ère fois) : ${mb} / ${Math.round(total / 1048576)} Mo`, (loaded / total) * 0.25);
        }
    }
    const bytes = new Uint8Array(loaded);
    let off = 0;
    for (const c of chunks) { bytes.set(c, off); off += c.length; }

    status(id, "Contrôle d'intégrité SHA-256 de l'IA...", 0.28);
    await verifyAndGetBuffer(bytes.buffer);

    if (cache) {
        try {
            await cache.put(MODEL_URL, new Response(bytes, { headers: { 'Content-Type': 'application/octet-stream' } }));
        } catch (e) {
            console.warn('[Voksplit Worker] Mise en cache impossible :', e);
        }
    }
    return bytes.buffer;
}

function getProcessor(id) {
    if (processorPromise) return processorPromise;
    processorPromise = (async () => {
        const ort = await import(`${ORT_BASE}ort.webgpu.min.mjs`);
        ort.env.wasm.wasmPaths = ORT_BASE;
        ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(navigator.hardwareConcurrency || 4, 8) : 1;
        ort.env.wasm.proxy = false;

        const { DemucsProcessor } = await import(DEMUCS_LOCAL_PATH);

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

        const processor = new DemucsProcessor({
            ort,
            sessionOptions: { executionProviders: providers, enableCpuMemArena: false, enableMemPattern: false },
            onProgress: (info) => segmentCb(info),
            onLog: (phase, msg) => console.log(`[Voksplit Demucs][${phase}] ${msg}`)
        });

        const modelBuffer = await getModelBuffer(id);
        const initMsg = hasWebGpu ? "Initialisation de l'IA (GPU ultra-rapide ⚡)..." : "Initialisation de l'IA (Mode CPU multicœur)...";
        status(id, initMsg, 0.32);
        await processor.loadModel(modelBuffer);
        return processor;
    })();
    processorPromise.catch(() => { processorPromise = null; });
    return processorPromise;
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
        pcm16[p++] = Math.max(-32768, Math.min(32767, Math.round(sL * (sL < 0 ? 32768 : 32767))));

        let sR = right[i];
        if (isNaN(sR) || !isFinite(sR)) sR = 0;
        else if (sR > 0.999) sR = 0.999;
        else if (sR < -0.999) sR = -0.999;
        pcm16[p++] = Math.max(-32768, Math.min(32767, Math.round(sR * (sR < 0 ? 32768 : 32767))));
    }
    return new Blob([buffer], { type: 'audio/wav' });
}

self.onmessage = async (e) => {
    const { id, left, right } = e.data || {};
    try {
        status(id, "Préparation de l'IA locale...", 0.05);
        const processor = await getProcessor(id);

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
            let dL = stems.drums?.left?.[i] || 0;
            let dR = stems.drums?.right?.[i] || 0;
            let bL = stems.bass?.left?.[i] || 0;
            let bR = stems.bass?.right?.[i] || 0;
            let oL = stems.other?.left?.[i] || 0;
            let oR = stems.other?.right?.[i] || 0;

            let sL = dL + bL + oL;
            let sR = dR + bR + oR;
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

        // Limiteur transparent soft tanh
        for (let i = 0; i < n; i++) {
            instL[i] = Math.tanh(instL[i]);
            instR[i] = Math.tanh(instR[i]);
            vocL[i] = Math.tanh(vocL[i]);
            vocR[i] = Math.tanh(vocR[i]);
        }

        const instrumentalBlob = encodeWav(instL, instR);
        const vocalBlob = encodeWav(vocL, vocR);

        self.postMessage({ id, type: 'done', instrumentalBlob, vocalBlob });
    } catch (err) {
        self.postMessage({ id, type: 'error', message: err && err.message ? err.message : String(err) });
    }
};
