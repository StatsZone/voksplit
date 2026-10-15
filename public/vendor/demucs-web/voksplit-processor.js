/**
 * VoksplitProcessor - Moteur d'inférence haute performance pour VoksplitNet
 * 100% Client-Side WebGPU / WASM SIMD Multithread
 * Architecture : TFC-TDF U-Net (2 stems : Voix & Instrumental)
 */

import { stft, istft, getHannWindow } from './fft.js';

export class VoksplitProcessor {
  constructor(options = {}) {
    this.ort = options.ort;
    this.session = null;
    this.sessionOptions = options.sessionOptions || {};
    this.onProgress = options.onProgress || (() => {});
    this.onLog = options.onLog || (() => {});

    this.sampleRate = 44100;
    this.fftSize = 2048;
    this.hopSize = 512;
    this.numBins = 1024; // 1024 bacs de fréquence (sur 1025)
    // 512 trames exactes pour correspondre aux 3 niveaux de pooling x2 du U-Net (2^3 = 8)
    // Formule STFT : (512 - 1) * 512 + 2048 = 263680 échantillons (~5.98 secondes)
    this.segmentSamples = 263680;
    this.overlapRatio = 0.5; // 50% de recouvrement pour un fondu enchaîné parfait
  }

  async loadModel(modelBuffer) {
    if (!this.ort) throw new Error("ONNX Runtime Web n'est pas initialisé.");
    this.onLog('load', 'Chargement du modèle VoksplitNet dans ONNX Runtime...');
    this.session = await this.ort.InferenceSession.create(modelBuffer, this.sessionOptions);
    this.onLog('load', 'Modèle VoksplitNet initialisé avec succès.');
  }

  /**
   * Séparation stéréo Voix / Instrumental
   * @param {Float32Array} left - Canal gauche
   * @param {Float32Array} right - Canal droit
   * @returns {Promise<{vocals: {left: Float32Array, right: Float32Array}, instrumental: {left: Float32Array, right: Float32Array}}>}
   */
  async separate(left, right) {
    if (!this.session) throw new Error("Le modèle VoksplitNet n'est pas chargé.");

    const totalLength = left.length;
    const stepSamples = Math.floor(this.segmentSamples * (1 - this.overlapRatio));
    const totalSegments = Math.max(1, Math.ceil((totalLength - this.segmentSamples) / stepSamples) + 1);

    const vocAccumL = new Float32Array(totalLength);
    const vocAccumR = new Float32Array(totalLength);
    const weightAccum = new Float32Array(totalLength);

    const hannWindow = getHannWindow(this.segmentSamples);

    for (let segIdx = 0; segIdx < totalSegments; segIdx++) {
      const start = segIdx * stepSamples;
      const end = Math.min(start + this.segmentSamples, totalLength);
      const chunkLen = end - start;

      // Découpe et padding si fin de morceau
      const chunkL = new Float32Array(this.segmentSamples);
      const chunkR = new Float32Array(this.segmentSamples);
      chunkL.set(left.subarray(start, end));
      chunkR.set(right.subarray(start, end));

      // 1. Calcul STFT pour gauche et droite
      const stftL = stft(chunkL, this.fftSize, this.hopSize);
      const stftR = stft(chunkR, this.fftSize, this.hopSize);
      // Garantir strictement un multiple de 8 pour les skip-connections du décodeur U-Net
      const numFrames = Math.floor(stftL.numFrames / 8) * 8;
      const numBinsTotal = stftL.numBins; // 1025

      // 2. Calcul des magnitudes et préparation du tenseur [1, 2, 1024, numFrames]
      const tensorData = new Float32Array(1 * 2 * this.numBins * numFrames);
      const magL = new Float32Array(this.numBins * numFrames);
      const magR = new Float32Array(this.numBins * numFrames);

      const offsetCh0 = 0;
      const offsetCh1 = this.numBins * numFrames;

      for (let f = 0; f < numFrames; f++) {
        for (let b = 0; b < this.numBins; b++) {
          const specIdx = f * numBinsTotal + b;
          const tensorIdx = b * numFrames + f;

          const rL = stftL.real[specIdx];
          const iL = stftL.imag[specIdx];
          const mL = Math.sqrt(rL * rL + iL * iL);
          magL[tensorIdx] = mL;
          tensorData[offsetCh0 + tensorIdx] = mL;

          const rR = stftR.real[specIdx];
          const iR = stftR.imag[specIdx];
          const mR = Math.sqrt(rR * rR + iR * iR);
          magR[tensorIdx] = mR;
          tensorData[offsetCh1 + tensorIdx] = mR;
        }
      }

      // 3. Exécution de l'inférence neuronale
      const inputTensor = new this.ort.Tensor('float32', tensorData, [1, 2, this.numBins, numFrames]);
      const outputs = await this.session.run({ mix_mag: inputTensor });

      const maskVocData = outputs.mask_vocals ? outputs.mask_vocals.data : null;
      if (!maskVocData) throw new Error("Sortie de masque vocal manquante dans le modèle.");

      // 4. Application du masque vocal et reconstruction spectrale
      const vocSpecRealL = new Float32Array(numFrames * numBinsTotal);
      const vocSpecImagL = new Float32Array(numFrames * numBinsTotal);
      const vocSpecRealR = new Float32Array(numFrames * numBinsTotal);
      const vocSpecImagR = new Float32Array(numFrames * numBinsTotal);

      for (let f = 0; f < numFrames; f++) {
        for (let b = 0; b < numBinsTotal; b++) {
          const specIdx = f * numBinsTotal + b;
          if (b < this.numBins) {
            const tensorIdx = b * numFrames + f;
            const mVocL = Math.max(0, Math.min(1, maskVocData[offsetCh0 + tensorIdx]));
            const mVocR = Math.max(0, Math.min(1, maskVocData[offsetCh1 + tensorIdx]));

            vocSpecRealL[specIdx] = stftL.real[specIdx] * mVocL;
            vocSpecImagL[specIdx] = stftL.imag[specIdx] * mVocL;
            vocSpecRealR[specIdx] = stftR.real[specIdx] * mVocR;
            vocSpecImagR[specIdx] = stftR.imag[specIdx] * mVocR;
          } else {
            // Dernier bac (1024ème / Nyquist)
            vocSpecRealL[specIdx] = 0;
            vocSpecImagL[specIdx] = 0;
            vocSpecRealR[specIdx] = 0;
            vocSpecImagR[specIdx] = 0;
          }
        }
      }

      // 5. Synthèse temporelle inverse (iSTFT)
      const vocTimeL = istft(vocSpecRealL, vocSpecImagL, numFrames, numBinsTotal, this.fftSize, this.hopSize, this.segmentSamples);
      const vocTimeR = istft(vocSpecRealR, vocSpecImagR, numFrames, numBinsTotal, this.fftSize, this.hopSize, this.segmentSamples);

      // 6. Accumulation avec fenêtre de lissage (Crossfade Hann)
      for (let i = 0; i < chunkLen; i++) {
        const w = hannWindow[i] || 1;
        vocAccumL[start + i] += vocTimeL[i] * w;
        vocAccumR[start + i] += vocTimeR[i] * w;
        weightAccum[start + i] += w;
      }

      this.onProgress({ currentSegment: segIdx + 1, totalSegments });
    }

    // 7. Normalisation par les poids et calcul de l'instru par soustraction de phase transparente
    const finalVocL = new Float32Array(totalLength);
    const finalVocR = new Float32Array(totalLength);
    const finalInsL = new Float32Array(totalLength);
    const finalInsR = new Float32Array(totalLength);

    for (let i = 0; i < totalLength; i++) {
      const w = weightAccum[i] > 1e-6 ? weightAccum[i] : 1;
      const vL = vocAccumL[i] / w;
      const vR = vocAccumR[i] / w;

      finalVocL[i] = vL;
      finalVocR[i] = vR;

      // Méthode studio : Instru = Mix Original - Voix Isolée
      // Préserve 100% de la dynamique des batteries, basses et synthés sans aucun artefact d'inversion de phase
      finalInsL[i] = left[i] - vL;
      finalInsR[i] = right[i] - vR;
    }

    // 8. Normalisation anti-saturation transparente (-0.5 dB)
    let peakIns = 0;
    let peakVoc = 0;
    for (let i = 0; i < totalLength; i++) {
      const absIL = Math.abs(finalInsL[i]);
      const absIR = Math.abs(finalInsR[i]);
      if (absIL > peakIns) peakIns = absIL;
      if (absIR > peakIns) peakIns = absIR;

      const absVL = Math.abs(finalVocL[i]);
      const absVR = Math.abs(finalVocR[i]);
      if (absVL > peakVoc) peakVoc = absVL;
      if (absVR > peakVoc) peakVoc = absVR;
    }

    if (peakIns > 0.95) {
      const s = 0.95 / peakIns;
      for (let i = 0; i < totalLength; i++) {
        finalInsL[i] *= s;
        finalInsR[i] *= s;
      }
    }

    if (peakVoc > 0.95) {
      const s = 0.95 / peakVoc;
      for (let i = 0; i < totalLength; i++) {
        finalVocL[i] *= s;
        finalVocR[i] *= s;
      }
    }

    return {
      vocals: { left: finalVocL, right: finalVocR },
      instrumental: { left: finalInsL, right: finalInsR }
    };
  }
}
