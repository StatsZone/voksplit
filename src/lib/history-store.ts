// src/lib/history-store.ts - Stockage local IndexedDB des morceaux et stems isolés pour Voksplit

import { openDB, type DBSchema, type IDBPDatabase } from 'idb';

export interface AudioItem {
  id: string;
  title: string;
  fileName: string;
  originalBlob?: Blob;
  instrumentalBlob?: Blob;
  vocalsBlob?: Blob;
  duration: number;
  engine: 'ai' | 'dsp';
  createdAt: number;
}

interface VoksplitDB extends DBSchema {
  tracks: {
    key: string;
    value: AudioItem;
    indexes: { 'by-date': number };
  };
}

let dbPromise: Promise<IDBPDatabase<VoksplitDB>> | null = null;

function getDB() {
  if (!dbPromise) {
    dbPromise = openDB<VoksplitDB>('voksplit-audio-db', 1, {
      upgrade(db) {
        const store = db.createObjectStore('tracks', { keyPath: 'id' });
        store.createIndex('by-date', 'createdAt');
      },
    });
  }
  return dbPromise;
}

export async function saveAudioItem(item: AudioItem): Promise<void> {
  const db = await getDB();
  await db.put('tracks', item);
}

export async function getAllAudioItems(): Promise<AudioItem[]> {
  const db = await getDB();
  const all = await db.getAllFromIndex('tracks', 'by-date');
  return all.reverse();
}

export async function deleteAudioItem(id: string): Promise<void> {
  const db = await getDB();
  await db.delete('tracks', id);
}
