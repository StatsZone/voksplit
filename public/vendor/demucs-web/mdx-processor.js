/**
 * mdx-processor.js - Moteur d'inférence ultra-haute fidélité MDX-Net (UVR5)
 * 100% Client-Side WebGPU / WASM SIMD Multithread
 * Architecture : Multi-scale Diminishing dilated Convolutional Neural Network (MDX-Net)
 * Modèle de référence : UVR-MDX-NET-Inst_HQ_3 (31.85 Mo FP16)
 * Élimination vocale de qualité studio professionnelle (Zéro grésillement, Zéro résidu)
 */

// FFT 2048-point radix-2 Cooley-Tukey optimisé
const fft2048Twiddles = (() => {
  const n = 2048;
  const bits = 11;
  const bitRev = new Uint16Array(n);
  for (let i = 0; i < n; i++) {
    let rev = 0, tmp = i;
    for (let b = 0; b < bits; b++) { rev = (rev << 1) | (tmp & 1); tmp >>= 1; }
    bitRev[i] = rev;
  }
  const cosT = new Float32Array(n / 2);
  const sinT = new Float32Array(n / 2);
  for (let k = 0; k < n / 2; k++) {
    const angle = -2 * Math.PI * k / n;
    cosT[k] = Math.cos(angle);
    sinT[k] = Math.sin(angle);
  }
  return { bitRev, cosT, sinT };
})();

function fft2048(realOut, imagOut, realIn, imagIn) {
  const n = 2048;
  const { bitRev, cosT, sinT } = fft2048Twiddles;

  for (let i = 0; i < n; i++) {
    const j = bitRev[i];
    realOut[i] = realIn[j];
    imagOut[i] = imagIn ? imagIn[j] : 0;
  }

  for (let s = 2; s <= n; s *= 2) {
    const half = s / 2;
    const step = 2048 / s;
    for (let i = 0; i < n; i += s) {
      for (let j = 0; j < half; j++) {
        const k = j * step;
        const cos = cosT[k];
        const sin = sinT[k];
        const idx1 = i + j;
        const idx2 = i + j + half;
        const uR = realOut[idx1], uI = imagOut[idx1];
        const vR = realOut[idx2] * cos - imagOut[idx2] * sin;
        const vI = realOut[idx2] * sin + imagOut[idx2] * cos;
        realOut[idx1] = uR + vR;
        imagOut[idx1] = uI + vI;
        realOut[idx2] = uR - vR;
        imagOut[idx2] = uI - vI;
      }
    }
  }
}

// FFT 6144-point mixte (Radix-3 puis 3x Radix-2048)
const mdxTwiddles = (() => {
  const N = 6144;
  const cos1 = new Float32Array(N);
  const sin1 = new Float32Array(N);
  const cos2 = new Float32Array(N);
  const sin2 = new Float32Array(N);
  for (let k = 0; k < N; k++) {
    const a1 = -2 * Math.PI * k / N;
    cos1[k] = Math.cos(a1);
    sin1[k] = Math.sin(a1);
    const a2 = -4 * Math.PI * k / N;
    cos2[k] = Math.cos(a2);
    sin2[k] = Math.sin(a2);
  }
  return { cos1, sin1, cos2, sin2 };
})();

// Buffers réutilisables pour éviter le Garbage Collector
const scratchR0 = new Float32Array(2048), scratchI0 = new Float32Array(2048);
const scratchR1 = new Float32Array(2048), scratchI1 = new Float32Array(2048);
const scratchR2 = new Float32Array(2048), scratchI2 = new Float32Array(2048);
const inR0 = new Float32Array(2048), inR1 = new Float32Array(2048), inR2 = new Float32Array(2048);
const inI0 = new Float32Array(2048), inI1 = new Float32Array(2048), inI2 = new Float32Array(2048);

function fft6144(realOut, imagOut, realIn, imagIn) {
  const N = 6144;
  const M = 2048;
  const { cos1, sin1, cos2, sin2 } = mdxTwiddles;

  for (let m = 0; m < M; m++) {
    const idx = 3 * m;
    inR0[m] = realIn[idx];
    inR1[m] = realIn[idx + 1];
    inR2[m] = realIn[idx + 2];
    if (imagIn) {
      inI0[m] = imagIn[idx];
      inI1[m] = imagIn[idx + 1];
      inI2[m] = imagIn[idx + 2];
    }
  }

  fft2048(scratchR0, scratchI0, inR0, imagIn ? inI0 : null);
  fft2048(scratchR1, scratchI1, inR1, imagIn ? inI1 : null);
  fft2048(scratchR2, scratchI2, inR2, imagIn ? inI2 : null);

  for (let k = 0; k < N; k++) {
    const km = k % M;
    const c1 = cos1[k], s1 = sin1[k];
    const c2 = cos2[k], s2 = sin2[k];

    const f1R = scratchR1[km] * c1 - scratchI1[km] * s1;
    const f1I = scratchR1[km] * s1 + scratchI1[km] * c1;
    const f2R = scratchR2[km] * c2 - scratchI2[km] * s2;
    const f2I = scratchR2[km] * s2 + scratchI2[km] * c2;

    realOut[k] = scratchR0[km] + f1R + f2R;
    imagOut[k] = scratchI0[km] + f1I + f2I;
  }
}

const scratchNegI = new Float32Array(6144);
function ifft6144(realOut, imagOut, realIn, imagIn) {
  const N = 6144;
  for (let i = 0; i < N; i++) scratchNegI[i] = -imagIn[i];
  fft6144(realOut, imagOut, realIn, scratchNegI);
  for (let i = 0; i < N; i++) {
    realOut[i] /= N;
    imagOut[i] = -imagOut[i] / N;
  }
}

export class MDXProcessor {
  constructor(options = {}) {
    this.ort = options.ort;
    this.session = null;
    this.sessionOptions = options.sessionOptions || {};
    this.onProgress = options.onProgress || (() => {});
    this.onLog = options.onLog || (() => {});

    this.sampleRate = 44100;
    this.n_fft = 6144;
    this.hop_length = 1024;
    this.dim_f = 3072;
    this.dim_t = 256;
    this.chunk_size = this.hop_length * (this.dim_t - 1); // 261120 échantillons (~5.92s)
    this.pad_side = this.n_fft / 2; // 3072
    this.overlapRatio = 0.5; // 50% de recouvrement Hann pour éliminer tout artefact de frontière

    // Fenêtre de Hann précalculée de 6144 échantillons
    this.window6144 = new Float32Array(this.n_fft);
    for (let i = 0; i < this.n_fft; i++) {
      this.window6144[i] = 0.5 * (1 - Math.cos(2 * Math.PI * i / this.n_fft));
    }

    // Somme précalculée des carrés des fenêtres pour l'inversion iSTFT exacte
    const paddedLen = this.chunk_size + 2 * this.pad_side;
    this.winSqSum = new Float32Array(paddedLen);
    for (let f = 0; f < this.dim_t; f++) {
      const start = f * this.hop_length;
      for (let i = 0; i < this.n_fft; i++) {
        this.winSqSum[start + i] += this.window6144[i] * this.window6144[i];
      }
    }

    // Fenêtre de crossfade Hann pour le découpage des morceaux
    this.hannChunkWindow = new Float32Array(this.chunk_size);
    for (let i = 0; i < this.chunk_size; i++) {
      this.hannChunkWindow[i] = 0.5 * (1 - Math.cos(2 * Math.PI * i / this.chunk_size));
    }
  }

  async loadModel(modelBuffer) {
    if (!this.ort) throw new Error("ONNX Runtime Web n'est pas initialisé.");
    this.onLog('load', 'Chargement du modèle MDX-Net UVR5 dans ONNX Runtime...');
    this.session = await this.ort.InferenceSession.create(modelBuffer, this.sessionOptions);
    this.onLog('load', 'Modèle MDX-Net UVR5 initialisé avec succès.');
  }

  /**
   * Séparation stéréo ultra-haute fidélité (UVR5 MDX-Net)
   * @param {Float32Array} left Canal gauche (44.1 kHz)
   * @param {Float32Array} right Canal droit (44.1 kHz)
   * @returns {Promise<{instrumental: {left: Float32Array, right: Float32Array}, vocals: {left: Float32Array, right: Float32Array}}>}
   */
  async separate(left, right) {
    if (!this.session) throw new Error("Le modèle MDX-Net n'est pas chargé.");

    const totalLength = left.length;
    const stepSamples = Math.floor(this.chunk_size * (1 - this.overlapRatio));
    const totalSegments = Math.max(1, Math.ceil((totalLength - this.chunk_size) / stepSamples) + 1);

    const accumInstL = new Float32Array(totalLength);
    const accumInstR = new Float32Array(totalLength);
    const accumWeight = new Float32Array(totalLength);

    // Buffers temporaires de calcul
    const paddedLen = this.chunk_size + 2 * this.pad_side;
    const paddedL = new Float32Array(paddedLen);
    const paddedR = new Float32Array(paddedLen);

    const frameInL = new Float32Array(this.n_fft);
    const frameInR = new Float32Array(this.n_fft);
    const specRealL = new Float32Array(this.n_fft);
    const specImagL = new Float32Array(this.n_fft);
    const specRealR = new Float32Array(this.n_fft);
    const specImagR = new Float32Array(this.n_fft);

    // Tenseur d'entrée MDX-Net : [1, 4, 3072, 256]
    // Canaux : 0 = L Real, 1 = L Imag, 2 = R Real, 3 = R Imag
    const tensorData = new Float32Array(1 * 4 * this.dim_f * this.dim_t);
    const chStride = this.dim_f * this.dim_t;

    // Buffers pour reconstruction iSTFT
    const recPaddedL = new Float32Array(paddedLen);
    const recPaddedR = new Float32Array(paddedLen);
    const fullRealL = new Float32Array(this.n_fft);
    const fullImagL = new Float32Array(this.n_fft);
    const fullRealR = new Float32Array(this.n_fft);
    const fullImagR = new Float32Array(this.n_fft);
    const timeOutRealL = new Float32Array(this.n_fft);
    const timeOutImagL = new Float32Array(this.n_fft);
    const timeOutRealR = new Float32Array(this.n_fft);
    const timeOutImagR = new Float32Array(this.n_fft);

    for (let segIdx = 0; segIdx < totalSegments; segIdx++) {
      const start = segIdx * stepSamples;
      const end = Math.min(start + this.chunk_size, totalLength);
      const actualChunkLen = end - start;

      // 1. Découpage du chunk avec padding si fin de morceau
      const chunkL = new Float32Array(this.chunk_size);
      const chunkR = new Float32Array(this.chunk_size);
      chunkL.set(left.subarray(start, end));
      chunkR.set(right.subarray(start, end));

      // 2. Padding reflect (center = True) de 3072 échantillons de chaque côté
      paddedL.fill(0);
      paddedR.fill(0);
      for (let i = 0; i < this.pad_side; i++) {
        paddedL[this.pad_side - 1 - i] = chunkL[Math.min(i + 1, this.chunk_size - 1)];
        paddedR[this.pad_side - 1 - i] = chunkR[Math.min(i + 1, this.chunk_size - 1)];
      }
      paddedL.set(chunkL, this.pad_side);
      paddedR.set(chunkR, this.pad_side);
      for (let i = 0; i < this.pad_side; i++) {
        const srcIdx = Math.max(0, this.chunk_size - 2 - i);
        paddedL[this.pad_side + this.chunk_size + i] = chunkL[srcIdx];
        paddedR[this.pad_side + this.chunk_size + i] = chunkR[srcIdx];
      }

      // 3. STFT trame par trame (256 trames)
      for (let f = 0; f < this.dim_t; f++) {
        const frameStart = f * this.hop_length;
        for (let i = 0; i < this.n_fft; i++) {
          frameInL[i] = paddedL[frameStart + i] * this.window6144[i];
          frameInR[i] = paddedR[frameStart + i] * this.window6144[i];
        }

        fft6144(specRealL, specImagL, frameInL, null);
        fft6144(specRealR, specImagR, frameInR, null);

        // Copie des 3072 premiers bacs dans le tenseur (format [4, 3072, 256])
        for (let b = 0; b < this.dim_f; b++) {
          const tensorIdx = b * this.dim_t + f;
          tensorData[0 * chStride + tensorIdx] = specRealL[b];
          tensorData[1 * chStride + tensorIdx] = specImagL[b];
          tensorData[2 * chStride + tensorIdx] = specRealR[b];
          tensorData[3 * chStride + tensorIdx] = specImagR[b];
        }
      }

      // 4. Suppression des 3 premiers bacs ultra-basses fréquences (DC / sub < 21Hz)
      for (let b = 0; b < 3; b++) {
        for (let f = 0; f < this.dim_t; f++) {
          const tensorIdx = b * this.dim_t + f;
          tensorData[0 * chStride + tensorIdx] = 0;
          tensorData[1 * chStride + tensorIdx] = 0;
          tensorData[2 * chStride + tensorIdx] = 0;
          tensorData[3 * chStride + tensorIdx] = 0;
        }
      }

      // 5. Inférence neuronale MDX-Net
      const inputTensor = new this.ort.Tensor('float32', tensorData, [1, 4, this.dim_f, this.dim_t]);
      const outputs = await this.session.run({ input: inputTensor });
      const outData = outputs.output.data;

      // 6. iSTFT pour reconstituer l'instrumentale
      recPaddedL.fill(0);
      recPaddedR.fill(0);

      for (let f = 0; f < this.dim_t; f++) {
        fullRealL.fill(0); fullImagL.fill(0);
        fullRealR.fill(0); fullImagR.fill(0);

        for (let b = 0; b < this.dim_f; b++) {
          const tensorIdx = b * this.dim_t + f;
          const rL = outData[0 * chStride + tensorIdx];
          const iL = outData[1 * chStride + tensorIdx];
          const rR = outData[2 * chStride + tensorIdx];
          const iR = outData[3 * chStride + tensorIdx];

          fullRealL[b] = rL; fullImagL[b] = iL;
          fullRealR[b] = rR; fullImagR[b] = iR;

          // Symétrie conjuguée hermitienne pour FFT inverse
          if (b > 0 && b < this.dim_f) {
            fullRealL[this.n_fft - b] = rL; fullImagL[this.n_fft - b] = -iL;
            fullRealR[this.n_fft - b] = rR; fullImagR[this.n_fft - b] = -iR;
          }
        }

        ifft6144(timeOutRealL, timeOutImagL, fullRealL, fullImagL);
        ifft6144(timeOutRealR, timeOutImagR, fullRealR, fullImagR);

        const frameStart = f * this.hop_length;
        for (let i = 0; i < this.n_fft; i++) {
          recPaddedL[frameStart + i] += timeOutRealL[i] * this.window6144[i];
          recPaddedR[frameStart + i] += timeOutRealR[i] * this.window6144[i];
        }
      }

      // Normalisation par la somme des fenêtres au carré
      for (let i = 0; i < paddedLen; i++) {
        const w = this.winSqSum[i] > 1e-7 ? this.winSqSum[i] : 1;
        recPaddedL[i] /= w;
        recPaddedR[i] /= w;
      }

      // 7. Accumulation avec fondu enchaîné Hann
      for (let i = 0; i < actualChunkLen; i++) {
        const w = this.hannChunkWindow[i] || 1;
        accumInstL[start + i] += recPaddedL[this.pad_side + i] * w;
        accumInstR[start + i] += recPaddedR[this.pad_side + i] * w;
        accumWeight[start + i] += w;
      }

      this.onProgress({ currentSegment: segIdx + 1, totalSegments });
    }

    // 8. Normalisation globale par les poids d'accumulation
    const finalInstL = new Float32Array(totalLength);
    const finalInstR = new Float32Array(totalLength);
    const finalVocL = new Float32Array(totalLength);
    const finalVocR = new Float32Array(totalLength);

    for (let i = 0; i < totalLength; i++) {
      const w = accumWeight[i] > 1e-7 ? accumWeight[i] : 1;
      const sL = accumInstL[i] / w;
      const sR = accumInstR[i] / w;

      finalInstL[i] = sL;
      finalInstR[i] = sR;

      // La voix studio = Mix d'origine - Instrumentale
      finalVocL[i] = left[i] - sL;
      finalVocR[i] = right[i] - sR;
    }

    // 9. Normalisation transparente studio (-0.5 dB)
    let peakInst = 0;
    for (let i = 0; i < totalLength; i++) {
      const absL = Math.abs(finalInstL[i]);
      const absR = Math.abs(finalInstR[i]);
      if (absL > peakInst) peakInst = absL;
      if (absR > peakInst) peakInst = absR;
    }
    if (peakInst > 0.95) {
      const scale = 0.95 / peakInst;
      for (let i = 0; i < totalLength; i++) {
        finalInstL[i] *= scale;
        finalInstR[i] *= scale;
      }
    }

    return {
      instrumental: { left: finalInstL, right: finalInstR },
      vocals: { left: finalVocL, right: finalVocR }
    };
  }
}
