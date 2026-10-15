'use client';

import React, { useState, useRef, useEffect, useMemo } from 'react';
import { 
  Upload, 
  Sparkles, 
  Mic, 
  Music, 
  Play, 
  Pause, 
  Download, 
  Cpu, 
  ShieldCheck, 
  Layers, 
  Trash2, 
  RefreshCw,
  Sliders
} from 'lucide-react';
import { separateAudioAI, type SeparationResult } from '@/lib/audio-engine';
import { saveAudioItem, getAllAudioItems, deleteAudioItem, type AudioItem } from '@/lib/history-store';

export default function StudioPage() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [progressMsg, setProgressMsg] = useState('Prêt pour la séparation');
  const [progressPct, setProgressPct] = useState(0);

  // Résultats actuels
  const [currentResult, setCurrentResult] = useState<SeparationResult | null>(null);
  const [activeStem, setActiveStem] = useState<'orig' | 'instru' | 'vocals'>('orig');
  const [isPlaying, setIsPlaying] = useState(false);
  const [history, setHistory] = useState<AudioItem[]>([]);

  // Éléments audio
  const audioRefOrig = useRef<HTMLAudioElement | null>(null);
  const audioRefInstru = useRef<HTMLAudioElement | null>(null);
  const audioRefVocals = useRef<HTMLAudioElement | null>(null);

  // URLs Blob mémorisées et stables (évite toute réallocation en cours de lecture = zéro grésillement)
  const instruUrl = useMemo(() => {
    return currentResult?.instrumental ? URL.createObjectURL(currentResult.instrumental) : null;
  }, [currentResult?.instrumental]);

  const vocalsUrl = useMemo(() => {
    return currentResult?.vocals ? URL.createObjectURL(currentResult.vocals) : null;
  }, [currentResult?.vocals]);

  useEffect(() => {
    return () => {
      if (audioUrl) URL.revokeObjectURL(audioUrl);
      if (instruUrl) URL.revokeObjectURL(instruUrl);
      if (vocalsUrl) URL.revokeObjectURL(vocalsUrl);
    };
  }, [audioUrl, instruUrl, vocalsUrl]);

  // Charger l'historique local au montage
  useEffect(() => {
    loadHistory();
  }, []);

  async function loadHistory() {
    try {
      const items = await getAllAudioItems();
      setHistory(items);
    } catch (e) {
      console.warn('Historique non disponible:', e);
    }
  }

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      if (audioUrl) URL.revokeObjectURL(audioUrl);
      setSelectedFile(file);
      setAudioUrl(URL.createObjectURL(file));
      setCurrentResult(null);
      setActiveStem('orig');
      setIsPlaying(false);
      setProgressPct(0);
      setProgressMsg('Morceau chargé. Cliquez sur Séparer.');
    }
  };

  const handleStartSeparation = async () => {
    if (!selectedFile) return;

    setIsProcessing(true);
    setProgressPct(5);
    setProgressMsg('Initialisation...');

    try {
      const res = await separateAudioAI(selectedFile, ({ message, progress }) => {
        setProgressMsg(message);
        setProgressPct(Math.round(progress * 100));
      });

      setCurrentResult(res);
      setProgressPct(100);
      setProgressMsg('Séparation réussie ! Stems prêts.');

      // Sauvegarder dans l'historique IndexedDB
      const newItem: AudioItem = {
        id: 'track_' + Date.now(),
        title: selectedFile.name.replace(/\.[^.]+$/, ''),
        fileName: selectedFile.name,
        originalBlob: selectedFile,
        instrumentalBlob: res.instrumental,
        vocalsBlob: res.vocals,
        duration: 0,
        engine: 'ai',
        createdAt: Date.now(),
      };
      await saveAudioItem(newItem);
      await loadHistory();
    } catch (err: unknown) {
      console.error('Erreur séparation:', err);
      const msg = err instanceof Error ? err.message : String(err);
      setProgressMsg('Échec : ' + msg);
    } finally {
      setIsProcessing(false);
    }
  };

  const togglePlay = () => {
    const target = activeStem === 'instru' ? audioRefInstru.current : activeStem === 'vocals' ? audioRefVocals.current : audioRefOrig.current;
    if (!target) return;

    if (isPlaying) {
      audioRefOrig.current?.pause();
      audioRefInstru.current?.pause();
      audioRefVocals.current?.pause();
      setIsPlaying(false);
    } else {
      target.play().then(() => setIsPlaying(true)).catch(() => {});
    }
  };

  const switchStem = (stem: 'orig' | 'instru' | 'vocals') => {
    const currentAudio = activeStem === 'instru' ? audioRefInstru.current : activeStem === 'vocals' ? audioRefVocals.current : audioRefOrig.current;
    const currentTime = currentAudio ? currentAudio.currentTime : 0;
    const wasPlaying = isPlaying;

    if (currentAudio) currentAudio.pause();

    setActiveStem(stem);

    setTimeout(() => {
      const nextAudio = stem === 'instru' ? audioRefInstru.current : stem === 'vocals' ? audioRefVocals.current : audioRefOrig.current;
      if (nextAudio) {
        nextAudio.currentTime = currentTime;
        if (wasPlaying) {
          nextAudio.play().then(() => setIsPlaying(true)).catch(() => {});
        }
      }
    }, 20);
  };

  const downloadStem = (blob: Blob | undefined, suffix: string) => {
    if (!blob) return;
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${selectedFile ? selectedFile.name.replace(/\.[^.]+$/, '') : 'audio'}_${suffix}.wav`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="flex-1 flex flex-col bg-slate-950">
      {/* Header Chic Voksplit */}
      <header className="border-b border-slate-800/80 bg-slate-900/50 backdrop-blur-md sticky top-0 z-40 px-6 py-4 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 rounded-xl bg-gradient-to-tr from-cyan-500 to-indigo-600 flex items-center justify-center font-bold text-white shadow-lg shadow-cyan-500/20">
            VK
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h1 className="font-extrabold tracking-tight text-lg text-white">Voksplit</h1>
              <span className="text-xs px-2 py-0.5 rounded-full bg-cyan-950/80 border border-cyan-500/40 text-cyan-400 font-mono">
                by Sound Capsule
              </span>
            </div>
            <p className="text-xs text-slate-400">Séparation vocale IA 100% locale & privée</p>
          </div>
        </div>

        <div className="flex items-center gap-4 text-xs text-slate-400">
          <span className="flex items-center gap-1.5 bg-slate-800/60 px-3 py-1.5 rounded-lg border border-slate-700/50">
            <ShieldCheck className="w-4 h-4 text-emerald-400" />
            Zéro upload serveur
          </span>
          <span className="flex items-center gap-1.5 bg-slate-800/60 px-3 py-1.5 rounded-lg border border-slate-700/50">
            <Cpu className="w-4 h-4 text-cyan-400" />
            Accélération WebGPU
          </span>
        </div>
      </header>

      {/* Main Studio Area */}
      <main className="flex-1 max-w-6xl w-full mx-auto p-6 md:p-8 flex flex-col gap-8">
        
        {/* Upload & Setup Card */}
        <section className="bg-slate-900/40 border border-slate-800 rounded-2xl p-6 md:p-8 backdrop-blur shadow-xl flex flex-col gap-6">
          <div className="flex flex-col md:flex-row gap-6 items-start md:items-center justify-between">
            <div>
              <h2 className="text-xl font-bold text-white flex items-center gap-2">
                <Sparkles className="w-5 h-5 text-cyan-400" />
                Studio d&apos;Isolation Audio
              </h2>
              <p className="text-sm text-slate-400 mt-1">
                Formats acceptés : MP3, WAV, FLAC, M4A ou import direct depuis Sound Capsule Studio
              </p>
            </div>

          </div>

          {/* Drag & Drop Zone */}
          <label className="border-2 border-dashed border-slate-800 hover:border-cyan-500/50 transition-colors rounded-xl p-8 flex flex-col items-center justify-center gap-3 cursor-pointer bg-slate-950/40 group">
            <input
              type="file"
              accept="audio/*"
              className="hidden"
              onChange={handleFileChange}
              disabled={isProcessing}
            />
            <div className="w-14 h-14 rounded-2xl bg-cyan-950/40 border border-cyan-800/40 flex items-center justify-center text-cyan-400 group-hover:scale-105 transition-transform">
              <Upload className="w-6 h-6" />
            </div>
            <div className="text-center">
              <span className="font-semibold text-slate-200 block text-sm">
                {selectedFile ? selectedFile.name : 'Déposez votre chanson ici, ou cliquez pour parcourir'}
              </span>
              <span className="text-xs text-slate-500 mt-1 block">
                Traitement instantané et entièrement local sur votre appareil
              </span>
            </div>
          </label>

          {/* Action & Progress Bar */}
          {selectedFile && (
            <div className="flex flex-col gap-3 bg-slate-950/60 p-4 rounded-xl border border-slate-800">
              <div className="flex items-center justify-between text-xs">
                <span className="font-medium text-slate-300">{progressMsg}</span>
                <span className="font-mono text-cyan-400 font-bold">{progressPct}%</span>
              </div>
              <div className="w-full h-2 bg-slate-800 rounded-full overflow-hidden">
                <div
                  className="h-full bg-gradient-to-r from-cyan-500 to-indigo-500 transition-all duration-300 rounded-full"
                  style={{ width: `${progressPct}%` }}
                />
              </div>

              <div className="flex justify-end gap-3 mt-2">
                <button
                  type="button"
                  onClick={handleStartSeparation}
                  disabled={isProcessing}
                  className="flex items-center gap-2 px-5 py-2.5 rounded-xl font-bold text-sm bg-gradient-to-r from-cyan-400 to-indigo-500 hover:from-cyan-300 hover:to-indigo-400 text-slate-950 transition shadow-lg shadow-cyan-500/10 disabled:opacity-50 cursor-pointer"
                >
                  {isProcessing ? (
                    <>
                      <RefreshCw className="w-4 h-4 animate-spin" />
                      Traitement en cours...
                    </>
                  ) : (
                    <>
                      <Layers className="w-4 h-4" />
                      Séparer Voix & Instrumental
                    </>
                  )}
                </button>
              </div>
            </div>
          )}
        </section>

        {/* Player & Stems Comparison */}
        {currentResult && (
          <section className="bg-slate-900/40 border border-slate-800 rounded-2xl p-6 md:p-8 backdrop-blur shadow-xl flex flex-col gap-6">
            <div className="flex items-center justify-between">
              <div>
                <h3 className="text-lg font-bold text-white flex items-center gap-2">
                  <Sliders className="w-5 h-5 text-indigo-400" />
                  Lecteur Comparatif A/B Instantané
                </h3>
                <p className="text-xs text-slate-400">Basculez entre les pistes en temps réel sans latence</p>
              </div>

              <button
                type="button"
                onClick={togglePlay}
                className="w-12 h-12 rounded-full bg-cyan-400 hover:bg-cyan-300 text-slate-950 flex items-center justify-center transition shadow-lg shadow-cyan-400/20 cursor-pointer"
              >
                {isPlaying ? <Pause className="w-5 h-5" /> : <Play className="w-5 h-5 ml-0.5" />}
              </button>
            </div>

            {/* Stem Toggle Buttons */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <button
                type="button"
                onClick={() => switchStem('orig')}
                className={`p-4 rounded-xl border flex flex-col gap-2 transition text-left cursor-pointer ${
                  activeStem === 'orig'
                    ? 'border-cyan-500 bg-cyan-950/20 shadow-lg shadow-cyan-500/10'
                    : 'border-slate-800 bg-slate-950/40 hover:border-slate-700'
                }`}
              >
                <div className="flex items-center justify-between">
                  <span className="text-xs uppercase font-bold tracking-wider text-slate-400">Mix Original</span>
                  <Music className="w-4 h-4 text-slate-400" />
                </div>
                <span className="font-bold text-white text-base">Chanson Complète</span>
              </button>

              <button
                type="button"
                onClick={() => switchStem('instru')}
                className={`p-4 rounded-xl border flex flex-col gap-2 transition text-left cursor-pointer ${
                  activeStem === 'instru'
                    ? 'border-cyan-500 bg-cyan-950/20 shadow-lg shadow-cyan-500/10'
                    : 'border-slate-800 bg-slate-950/40 hover:border-slate-700'
                }`}
              >
                <div className="flex items-center justify-between">
                  <span className="text-xs uppercase font-bold tracking-wider text-indigo-400">Stem 1</span>
                  <Layers className="w-4 h-4 text-indigo-400" />
                </div>
                <div className="flex items-center justify-between">
                  <span className="font-bold text-white text-base">Instrumental Seul</span>
                  <button
                    type="button"
                    onClick={(e) => {
                      e.stopPropagation();
                      downloadStem(currentResult.instrumental, 'instru');
                    }}
                    className="p-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 transition"
                    title="Télécharger l'instrumentale WAV"
                  >
                    <Download className="w-4 h-4" />
                  </button>
                </div>
              </button>

              <button
                type="button"
                onClick={() => switchStem('vocals')}
                className={`p-4 rounded-xl border flex flex-col gap-2 transition text-left cursor-pointer ${
                  activeStem === 'vocals'
                    ? 'border-cyan-500 bg-cyan-950/20 shadow-lg shadow-cyan-500/10'
                    : 'border-slate-800 bg-slate-950/40 hover:border-slate-700'
                }`}
              >
                <div className="flex items-center justify-between">
                  <span className="text-xs uppercase font-bold tracking-wider text-rose-400">Stem 2</span>
                  <Mic className="w-4 h-4 text-rose-400" />
                </div>
                <div className="flex items-center justify-between">
                  <span className="font-bold text-white text-base">Voix Isolée</span>
                  <button
                    type="button"
                    onClick={(e) => {
                      e.stopPropagation();
                      downloadStem(currentResult.vocals, 'vocals');
                    }}
                    className="p-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 transition"
                    title="Télécharger la voix isolée WAV"
                  >
                    <Download className="w-4 h-4" />
                  </button>
                </div>
              </button>
            </div>

            {/* Hidden Audio Players for A/B Sync */}
            {audioUrl && <audio ref={audioRefOrig} src={audioUrl} onEnded={() => setIsPlaying(false)} className="hidden" />}
            {instruUrl && (
              <audio
                ref={audioRefInstru}
                src={instruUrl}
                onEnded={() => setIsPlaying(false)}
                className="hidden"
              />
            )}
            {vocalsUrl && (
              <audio
                ref={audioRefVocals}
                src={vocalsUrl}
                onEnded={() => setIsPlaying(false)}
                className="hidden"
              />
            )}
          </section>
        )}

        {/* Local History Section */}
        {history.length > 0 && (
          <section className="bg-slate-900/40 border border-slate-800 rounded-2xl p-6 backdrop-blur flex flex-col gap-4">
            <h3 className="text-base font-bold text-slate-300">Historique des Séparations (Stockage Local)</h3>
            <div className="divide-y divide-slate-800/80">
              {history.map((item) => (
                <div key={item.id} className="py-3 flex items-center justify-between gap-4">
                  <div className="flex items-center gap-3">
                    <div className="w-8 h-8 rounded-lg bg-slate-800 flex items-center justify-center text-slate-400">
                      <Music className="w-4 h-4" />
                    </div>
                    <div>
                      <span className="font-semibold text-white text-sm block">{item.title}</span>
                      <span className="text-xs text-slate-500">
                        {new Date(item.createdAt).toLocaleDateString('fr-FR')} • {item.engine === 'ai' ? 'IA Haute Fidélité' : 'Aperçu DSP'}
                      </span>
                    </div>
                  </div>

                  <div className="flex items-center gap-2">
                    {item.instrumentalBlob && (
                      <button
                        type="button"
                        onClick={() => downloadStem(item.instrumentalBlob, 'instru')}
                        className="px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-xs font-medium text-slate-200 flex items-center gap-1.5"
                      >
                        <Download className="w-3.5 h-3.5" />
                        Instru
                      </button>
                    )}
                    {item.vocalsBlob && (
                      <button
                        type="button"
                        onClick={() => downloadStem(item.vocalsBlob, 'vocals')}
                        className="px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-xs font-medium text-slate-200 flex items-center gap-1.5"
                      >
                        <Download className="w-3.5 h-3.5" />
                        Voix
                      </button>
                    )}
                    <button
                      type="button"
                      onClick={async () => {
                        await deleteAudioItem(item.id);
                        await loadHistory();
                      }}
                      className="p-1.5 rounded-lg text-slate-500 hover:text-rose-400 transition"
                      title="Supprimer de l'historique"
                    >
                      <Trash2 className="w-4 h-4" />
                    </button>
                  </div>
                </div>
              ))}
            </div>
          </section>
        )}
      </main>
    </div>
  );
}
