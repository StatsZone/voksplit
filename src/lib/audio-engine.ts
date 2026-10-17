// src/lib/audio-engine.ts - Moteur audio TypeScript pour Voksplit
// Décodage multi-formats, rééchantillonnage 44.1kHz, filtres DSP et pont Web Worker

const SAMPLE_RATE = 44100;

export interface SeparationProgress {
  message: string;
  progress: number;
}

export interface SeparationResult {
  instrumental: Blob;
  vocals: Blob;
}

/** Convertit des tableaux PCM Float32 en Blob WAV 16-bit */
export function channelsToWavBlob(channels: Float32Array[], sampleRate = SAMPLE_RATE): Blob {
  const numChannels = channels.length;
  const frames = channels[0].length;
  const dataLength = frames * numChannels * 2;
  const buffer = new ArrayBuffer(44 + dataLength);
  const view = new DataView(buffer);

  const writeString = (offset: number, s: string) => {
    for (let i = 0; i < s.length; i++) view.setUint8(offset + i, s.charCodeAt(i));
  };

  writeString(0, 'RIFF');
  view.setUint32(4, 36 + dataLength, true);
  writeString(8, 'WAVE');
  writeString(12, 'fmt ');
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, numChannels, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * numChannels * 2, true);
  view.setUint16(32, numChannels * 2, true);
  view.setUint16(34, 16, true);
  writeString(36, 'data');
  view.setUint32(40, dataLength, true);

  // Analyse du pic maximal sur tous les canaux pour éviter tout grésillement
  let maxPeak = 0;
  for (let ch = 0; ch < numChannels; ch++) {
    for (let i = 0; i < frames; i++) {
      const a = Math.abs(channels[ch][i]);
      if (a > maxPeak) maxPeak = a;
    }
  }
  const gain = maxPeak > 0.95 ? 0.95 / maxPeak : 1.0;

  let offset = 44;
  for (let i = 0; i < frames; i++) {
    for (let ch = 0; ch < numChannels; ch++) {
      let s = (channels[ch][i] || 0) * gain;
      if (isNaN(s) || !isFinite(s)) s = 0;
      else if (s > 0.999) s = 0.999;
      else if (s < -0.999) s = -0.999;
      const pcm = Math.max(-32767, Math.min(32767, Math.round(s * 32767)));
      view.setInt16(offset, pcm, true);
      offset += 2;
    }
  }
  return new Blob([buffer], { type: 'audio/wav' });
}

/** Décode et rééchantillonne fidèlement n'importe quel fichier audio en 44.1 kHz */
export async function decodeAudioTo44k(fileOrBlob: Blob): Promise<{ left: Float32Array; right: Float32Array; duration: number }> {
  const arrayBuf = await fileOrBlob.arrayBuffer();
  const AudioContextClass = window.AudioContext || (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext;
  const tempCtx = new AudioContextClass();
  const rawBuffer = await tempCtx.decodeAudioData(arrayBuf);
  tempCtx.close().catch(() => {});

  if (rawBuffer.sampleRate === SAMPLE_RATE) {
    const left = rawBuffer.getChannelData(0);
    const right = rawBuffer.numberOfChannels > 1 ? rawBuffer.getChannelData(1) : left;
    return { left, right, duration: rawBuffer.duration };
  }

  const targetLength = Math.max(1, Math.round(rawBuffer.duration * SAMPLE_RATE));
  const offlineCtx = new OfflineAudioContext(2, targetLength, SAMPLE_RATE);
  const srcNode = offlineCtx.createBufferSource();
  srcNode.buffer = rawBuffer;
  srcNode.connect(offlineCtx.destination);
  srcNode.start(0);

  const resampled = await offlineCtx.startRendering();
  const left = resampled.getChannelData(0);
  const right = resampled.getChannelData(1);
  return { left, right, duration: rawBuffer.duration };
}

/** Séparation IA Haute Fidélité via Web Worker non-bloquant */
export async function separateAudioAI(
  blob: Blob,
  onProgress?: (info: SeparationProgress) => void,
  engine: 'mdx' | 'voksplit' | 'demucs' = 'mdx'
): Promise<SeparationResult> {
  onProgress?.({ message: 'Décodage et normalisation audio...', progress: 0.05 });
  const { left, right } = await decodeAudioTo44k(blob);

  const lCopy = new Float32Array(left);
  const rCopy = new Float32Array(right);

  return new Promise((resolve, reject) => {
    const worker = new Worker('/stem-worker.js', { type: 'module' });
    const id = 'task_' + Date.now();

    worker.onmessage = (e) => {
      const data = e.data;
      if (!data || data.id !== id) return;

      if (data.type === 'status') {
        onProgress?.({ message: data.msg, progress: data.progress });
      } else if (data.type === 'done' || data.type === 'complete') {
        worker.terminate();
        resolve({
          instrumental: data.instrumentalBlob || data.blob,
          vocals: data.vocalsBlob || data.vocalBlob,
        });
      } else if (data.type === 'error') {
        worker.terminate();
        reject(new Error(data.message || data.error));
      }
    };

    worker.onerror = (err) => {
      worker.terminate();
      reject(new Error(err && err.message ? err.message : 'Erreur dans le Web Worker de séparation audio'));
    };

    worker.postMessage({ id, left: lCopy, right: rCopy, engine }, [lCopy.buffer, rCopy.buffer]);
  });
}

/** Filtre DSP rapide (Aperçu 1s - Mid/Side) */
export async function separateAudioDSP(
  blob: Blob,
  onProgress?: (info: SeparationProgress) => void
): Promise<SeparationResult> {
  onProgress?.({ message: 'Décodage studio...', progress: 0.15 });
  const { left, right } = await decodeAudioTo44k(blob);

  onProgress?.({ message: 'Filtrage spatial et extraction...', progress: 0.5 });
  const n = left.length;
  const outInstL = new Float32Array(n);
  const outInstR = new Float32Array(n);
  const outVocL = new Float32Array(n);
  const outVocR = new Float32Array(n);

  for (let i = 0; i < n; i++) {
    const sL = left[i] || 0;
    const sR = right[i] || 0;
    const mid = 0.5 * (sL + sR);
    const side = 0.5 * (sL - sR);

    // Instru : accentuation spatiale stéréo et atténuation douce du centre
    outInstL[i] = 0.25 * mid + side;
    outInstR[i] = 0.25 * mid - side;

    // Voix : extraction du centre spatial (canal mono centré)
    outVocL[i] = mid;
    outVocR[i] = mid;
  }

  onProgress?.({ message: 'Génération des fichiers...', progress: 0.95 });
  return {
    instrumental: channelsToWavBlob([outInstL, outInstR]),
    vocals: channelsToWavBlob([outVocL, outVocR]),
  };
}
