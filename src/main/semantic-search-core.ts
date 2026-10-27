/**
 * semantic-search-core.ts
 *
 * Pure helpers behind semantic-search.ts: text chunking, int8 vector
 * quantization and scoring. Deliberately dependency-free so tests can import
 * the real source (scripts/test-semantic-search-core.mjs).
 */

export const CHUNK_CHARS = 1200;
export const CHUNK_OVERLAP = 150;
export const MAX_CHUNKS_PER_ITEM = 8;

/**
 * Splits text into overlapping ~CHUNK_CHARS passages, preferring paragraph
 * then word boundaries, capped at MAX_CHUNKS_PER_ITEM (the start of a long
 * document is what gets embedded).
 */
export function chunkText(text: string): string[] {
  const clean = text
    .replace(/\r\n?/g, '\n')
    .replace(/[ \t\f\v]+/g, ' ')
    .replace(/\n{3,}/g, '\n\n')
    .trim();
  const chunks: string[] = [];
  let start = 0;
  while (start < clean.length && chunks.length < MAX_CHUNKS_PER_ITEM) {
    let end = Math.min(clean.length, start + CHUNK_CHARS);
    if (end < clean.length) {
      const minEnd = start + Math.floor(CHUNK_CHARS * 0.6);
      const paragraph = clean.lastIndexOf('\n', end);
      const space = clean.lastIndexOf(' ', end);
      if (paragraph > minEnd) end = paragraph;
      else if (space > minEnd) end = space;
    }
    const chunk = clean.slice(start, end).trim();
    if (chunk) chunks.push(chunk);
    if (end >= clean.length) break;
    // Step back for overlap, then forward to the next word start.
    let next = Math.max(end - CHUNK_OVERLAP, start + 1);
    while (next < end && !/\s/.test(clean[next - 1])) next++;
    start = next;
  }
  return chunks;
}

/** Quantizes unit vectors to int8 rows with a per-row scale (max |v| / 127). */
export function quantizeRows(vectors: ArrayLike<number>[], dim: number): { vectors: Int8Array; scales: Float32Array } {
  const out = new Int8Array(vectors.length * dim);
  const scales = new Float32Array(vectors.length);
  vectors.forEach((vector, row) => {
    let maxAbs = 0;
    for (let i = 0; i < dim; i++) maxAbs = Math.max(maxAbs, Math.abs(vector[i] || 0));
    const scale = maxAbs > 0 ? maxAbs / 127 : 1;
    scales[row] = scale;
    for (let i = 0; i < dim; i++) out[row * dim + i] = Math.round((vector[i] || 0) / scale);
  });
  return { vectors: out, scales };
}

/** Best cosine score (query · row) across an item's quantized rows. */
export function bestRowScore(
  query: ArrayLike<number>,
  vectors: Int8Array,
  scales: Float32Array,
  dim: number
): { score: number; row: number } {
  let score = -Infinity;
  let row = 0;
  for (let r = 0; r < scales.length; r++) {
    const base = r * dim;
    let dot = 0;
    for (let i = 0; i < dim; i++) dot += query[i] * vectors[base + i];
    dot *= scales[r];
    if (dot > score) {
      score = dot;
      row = r;
    }
  }
  return { score, row };
}

/** Small lexical nudge so exact filename/title hits win near-ties. */
export function titleBoost(title: string, terms: string[]): number {
  if (terms.length === 0) return 0;
  const lower = title.toLowerCase();
  const hits = terms.filter((t) => lower.includes(t)).length;
  return 0.04 * (hits / terms.length);
}

export function queryTerms(query: string): string[] {
  return query.toLowerCase().split(/\s+/).filter((t) => t.length >= 3);
}

/**
 * Removes most of the "average query" direction from a query vector.
 * Short, generic texts (clipboard snippets like "State" or a URL) sit close
 * to that direction and match every short query about equally well; without
 * it they crowd out real matches, especially across languages ("agreement"
 * vs Turkish "sözleşme" contracts).
 */
export function debiasQuery(query: ArrayLike<number>, centroid: ArrayLike<number>, strength: number): Float32Array {
  let along = 0;
  for (let i = 0; i < query.length; i++) along += query[i] * centroid[i];
  const out = new Float32Array(query.length);
  let norm = 0;
  for (let i = 0; i < query.length; i++) {
    out[i] = query[i] - strength * along * centroid[i];
    norm += out[i] * out[i];
  }
  norm = Math.sqrt(norm) || 1;
  for (let i = 0; i < out.length; i++) out[i] /= norm;
  return out;
}

/** Tuned on a real index: contracts for "agreement" went from 1/138/171 to 1/7/13 (1st/5th/10th). */
export const QUERY_DEBIAS_STRENGTH = 0.8;

/** Everyday searches whose mean embedding stands for "an average query". */
export const QUERY_CENTROID_PROBES = [
  'photo of a beach at sunset', 'invoice from last month', 'meeting notes about the budget', 'python script that parses csv',
  'recipe with chicken and rice', 'flight ticket to istanbul', 'screenshot of an error message', 'bank statement', 'kedi fotoğrafı',
  'toplantı notları', 'birthday party video', 'voice memo', 'tax documents', 'resume cv', 'presentation slides about marketing',
  'apartment rental', 'car insurance', 'doctor appointment', 'wifi password', 'api key', 'song lyrics', 'kitap özeti', 'fatura',
  'mountain hiking trip', 'wedding photos', 'project plan timeline', 'error log from the server', 'shopping list', 'email draft to my boss',
  'diagram of the architecture', 'phone number', 'contract',
];

/** Unit-length mean of a set of vectors. */
export function meanDirection(vectors: ArrayLike<number>[], dim: number): Float32Array {
  const out = new Float32Array(dim);
  for (const vector of vectors) {
    for (let i = 0; i < dim; i++) out[i] += vector[i];
  }
  let norm = 0;
  for (let i = 0; i < dim; i++) norm += out[i] * out[i];
  norm = Math.sqrt(norm) || 1;
  for (let i = 0; i < dim; i++) out[i] /= norm;
  return out;
}

/**
 * How far each score stands out from all the others (leave-one-out z-score,
 * as krunner-embed uses); the spread is floored so a tiny or uniform set
 * doesn't inflate it.
 */
export function standoutScores(scores: number[]): number[] {
  const n = scores.length;
  if (n < 3) return scores.map(() => 10);
  let sum = 0;
  let squares = 0;
  for (const s of scores) {
    sum += s;
    squares += s * s;
  }
  return scores.map((s) => {
    const mean = (sum - s) / (n - 1);
    const variance = (squares - s * s) / (n - 1) - mean * mean;
    return (s - mean) / Math.max(0.02, Math.sqrt(Math.max(0, variance)));
  });
}

// ─── Drive indexes ──────────────────────────────────────────────────
//
// Removable and network drives can carry their own index, in a format shared
// with krunner-embed on Linux (docs/semantic-index-format.md), so a drive
// indexed on one machine is searchable, and kept up to date, on the other.

export const DRIVE_INDEX_FOLDER = '.semantic-index.noindex';
export const DRIVE_INDEX_DIM = 256;
export const DRIVE_KINDS = ['document', 'image', 'video', 'audio'] as const;
export type DriveItemKind = (typeof DRIVE_KINDS)[number];

const DRIVE_INDEX_MAGIC = 'SEMINDEX';
const DRIVE_INDEX_VERSION = 1;
const DRIVE_INDEX_MODEL = 'embeddinggemma-2';
const DRIVE_INDEX_PROMPTS = 1;
const MAX_ROWS_PER_ITEM = 64;

export interface DriveIndexItem {
  /** Relative to the drive root, "/"-separated. */
  path: string;
  kind: DriveItemKind;
  /** ms since the epoch */
  mtime: number;
  size: number;
  hash: Uint8Array;
  scales: Float32Array;
  vectors: Int8Array;
  previews: string[];
}

export interface DriveIndexHeader {
  quant: string;
  lastIndexedAt: number;
  writer: string;
}

export type DecodedDriveIndex =
  | { ok: true; header: DriveIndexHeader; items: DriveIndexItem[] }
  | { ok: false; incompatible: boolean; error: string };

const utf8Encoder = new TextEncoder();
const utf8Decoder = new TextDecoder('utf-8');

function utf8Bytes(text: string, maxBytes: number): Uint8Array {
  const bytes = utf8Encoder.encode(text);
  if (bytes.length <= maxBytes) return bytes;
  let end = maxBytes;
  while (end > 0 && (bytes[end] & 0xc0) === 0x80) end--; // don't split a character
  return bytes.subarray(0, end);
}

/** A stored path must stay inside the drive. */
function isSafeRelativePath(relative: string): boolean {
  if (!relative || relative.startsWith('/')) return false;
  return relative.split('/').every((part) => part !== '' && part !== '..');
}

export function encodeDriveIndex(header: DriveIndexHeader, items: DriveIndexItem[]): Uint8Array {
  const dim = DRIVE_INDEX_DIM;
  const headerBytes = utf8Encoder.encode(JSON.stringify({
    model: DRIVE_INDEX_MODEL,
    quant: header.quant,
    dim,
    prompts: DRIVE_INDEX_PROMPTS,
    chunk: [CHUNK_CHARS, CHUNK_OVERLAP, MAX_CHUNKS_PER_ITEM],
    lastIndexedAt: header.lastIndexedAt,
    writer: header.writer,
  }));
  const stored = items
    .filter((item) => isSafeRelativePath(item.path) && item.scales.length <= MAX_ROWS_PER_ITEM)
    .map((item) => ({
      item,
      path: utf8Bytes(item.path, 0xffff),
      hash: item.hash.subarray(0, 255),
      previews: Array.from({ length: item.scales.length }, (_, r) => utf8Bytes(item.previews[r] || '', 0xffff)),
    }));
  let total = 8 + 4 + 4 + headerBytes.length + 4;
  for (const { item, path, hash, previews } of stored) {
    total += 2 + path.length + 1 + 1 + 8 + 8 + 1 + hash.length + 2 + item.scales.length * (4 + dim);
    for (const preview of previews) total += 2 + preview.length;
  }

  const out = new Uint8Array(total);
  const view = new DataView(out.buffer);
  let offset = 0;
  const bytes = (data: Uint8Array) => {
    out.set(data, offset);
    offset += data.length;
  };
  const u8 = (v: number) => { view.setUint8(offset, v); offset += 1; };
  const u16 = (v: number) => { view.setUint16(offset, v, true); offset += 2; };
  const u32 = (v: number) => { view.setUint32(offset, v, true); offset += 4; };
  const i64 = (v: number) => { view.setBigInt64(offset, BigInt(Math.trunc(v)), true); offset += 8; };

  bytes(utf8Encoder.encode(DRIVE_INDEX_MAGIC));
  u32(DRIVE_INDEX_VERSION);
  u32(headerBytes.length);
  bytes(headerBytes);
  u32(stored.length);
  for (const { item, path, hash, previews } of stored) {
    u16(path.length);
    bytes(path);
    u8(DRIVE_KINDS.indexOf(item.kind));
    u8(0);
    i64(item.mtime);
    i64(item.size);
    u8(hash.length);
    bytes(hash);
    u16(item.scales.length);
    for (const scale of item.scales) {
      view.setFloat32(offset, scale, true);
      offset += 4;
    }
    bytes(new Uint8Array(item.vectors.buffer, item.vectors.byteOffset, item.scales.length * dim));
    for (const preview of previews) {
      u16(preview.length);
      bytes(preview);
    }
  }
  return out;
}

export function decodeDriveIndex(data: Uint8Array): DecodedDriveIndex {
  const dim = DRIVE_INDEX_DIM;
  const view = new DataView(data.buffer, data.byteOffset, data.byteLength);
  let offset = 0;
  const need = (n: number) => {
    if (offset + n > data.length) throw new RangeError('truncated');
  };
  const take = (n: number) => {
    need(n);
    const slice = data.subarray(offset, offset + n);
    offset += n;
    return slice;
  };
  const u8 = () => { need(1); return view.getUint8(offset++); };
  const u16 = () => { need(2); const v = view.getUint16(offset, true); offset += 2; return v; };
  const u32 = () => { need(4); const v = view.getUint32(offset, true); offset += 4; return v; };
  const i64 = () => { need(8); const v = Number(view.getBigInt64(offset, true)); offset += 8; return v; };

  try {
    if (utf8Decoder.decode(take(8)) !== DRIVE_INDEX_MAGIC) return { ok: false, incompatible: false, error: 'not a semantic index' };
    if (u32() !== DRIVE_INDEX_VERSION) return { ok: false, incompatible: true, error: 'unsupported format version' };
    const header = JSON.parse(utf8Decoder.decode(take(u32())));
    const chunk = Array.isArray(header?.chunk) ? header.chunk : [];
    if (header?.model !== DRIVE_INDEX_MODEL || header?.dim !== dim || header?.prompts !== DRIVE_INDEX_PROMPTS
      || chunk[0] !== CHUNK_CHARS || chunk[1] !== CHUNK_OVERLAP || chunk[2] !== MAX_CHUNKS_PER_ITEM) {
      return { ok: false, incompatible: true, error: `made with ${header?.model || 'another model'}` };
    }
    const count = u32();
    const items: DriveIndexItem[] = [];
    for (let i = 0; i < count; i++) {
      const path = utf8Decoder.decode(take(u16()));
      const kind = u8();
      u8(); // flags
      const mtime = i64();
      const size = i64();
      const hash = take(u8()).slice();
      const rows = u16();
      if (kind >= DRIVE_KINDS.length || rows > MAX_ROWS_PER_ITEM) throw new RangeError('damaged item');
      const scales = new Float32Array(rows);
      need(rows * 4);
      for (let r = 0; r < rows; r++) {
        scales[r] = view.getFloat32(offset, true);
        offset += 4;
      }
      const vectors = new Int8Array(take(rows * dim).slice().buffer);
      const previews: string[] = [];
      for (let r = 0; r < rows; r++) previews.push(utf8Decoder.decode(take(u16())));
      if (isSafeRelativePath(path)) items.push({ path, kind: DRIVE_KINDS[kind], mtime, size, hash, scales, vectors, previews });
    }
    return {
      ok: true,
      header: {
        quant: String(header.quant || ''),
        lastIndexedAt: Number(header.lastIndexedAt) || 0,
        writer: String(header.writer || ''),
      },
      items,
    };
  } catch {
    return { ok: false, incompatible: false, error: 'truncated or damaged' };
  }
}

/**
 * Is a file unchanged, judging by its mtime (ms)? Exact, or with `tolerant`
 * the rule for drives: within 2 s, or a whole number of hours (up to 14)
 * ± 2 s apart — FAT stores local time with 2 s precision, so a time-zone
 * change shifts every timestamp.
 */
export function sameMtime(a: number, b: number, tolerant: boolean): boolean {
  if (a === b) return true;
  if (!tolerant) return false;
  const slack = 2000;
  const hour = 3600 * 1000;
  const diff = Math.abs(a - b);
  if (diff > 14 * hour + slack) return false;
  const offHour = diff % hour;
  return offHour <= slack || hour - offHour <= slack;
}

// Drive lock (lock.json): only its holder writes index.bin.

export const DRIVE_LOCK_STALE_MS = 15 * 60 * 1000;
const DRIVE_LOCK_FUTURE_MS = 24 * 3600 * 1000;

export interface DriveLockInfo {
  /** Parsed; an unparseable lock is judged by its file time. */
  valid: boolean;
  token: string;
  host: string;
  app: string;
  pid: number;
  since: number;
  heartbeat: number;
  fileTime: number;
}

export function parseDriveLock(text: string, fileTime: number): DriveLockInfo {
  let json: any = null;
  try {
    json = JSON.parse(text);
  } catch {}
  const heartbeat = Date.parse(String(json?.heartbeat ?? ''));
  const token = typeof json?.token === 'string' ? json.token : '';
  return {
    valid: Boolean(token) && Number.isFinite(heartbeat),
    token,
    host: typeof json?.host === 'string' ? json.host : '',
    app: typeof json?.app === 'string' ? json.app : '',
    pid: Number(json?.pid) || 0,
    since: Date.parse(String(json?.since ?? '')) || 0,
    heartbeat: Number.isFinite(heartbeat) ? heartbeat : 0,
    fileTime,
  };
}

export function driveLockOwner(lock: DriveLockInfo): string {
  if (!lock.valid) return 'an unknown app';
  return `${lock.app || 'an unknown app'} on ${lock.host}`;
}

/**
 * A lock may be taken over when its heartbeat is 15 minutes old (the holder
 * died or the drive was unplugged), or when it is from this machine and its
 * process is gone (or is our own pid: left over from before a restart).
 */
export function isDriveLockStale(
  lock: DriveLockInfo,
  now: number,
  thisHost: string,
  thisPid: number,
  isProcessAlive: (pid: number) => boolean
): boolean {
  if (!lock.valid) return !lock.fileTime || now - lock.fileTime > DRIVE_LOCK_STALE_MS;
  const age = now - lock.heartbeat;
  if (age > DRIVE_LOCK_STALE_MS || age < -DRIVE_LOCK_FUTURE_MS) return true;
  return lock.host === thisHost && (lock.pid === thisPid || !isProcessAlive(lock.pid));
}

// Mounted volumes (macOS `mount` output).

export interface MacMount {
  source: string;
  mountPoint: string;
  fsType: string;
  readOnly: boolean;
  /** `nobrowse`: hidden from Finder (system volumes such as Recovery). */
  hidden: boolean;
}

const NETWORK_FS_TYPES = new Set(['smbfs', 'nfs', 'afpfs', 'webdav', 'cifs', 'ftp']);

export function isNetworkFs(fsType: string): boolean {
  return NETWORK_FS_TYPES.has(fsType);
}

/** Parses lines like `/dev/disk4s1 on /Volumes/STICK (exfat, local, nodev, nosuid, read-only)`. */
export function parseMacMounts(text: string): MacMount[] {
  const mounts: MacMount[] = [];
  for (const line of text.split('\n')) {
    const on = line.indexOf(' on /');
    const open = line.lastIndexOf(' (');
    if (on < 0 || open < on || !line.endsWith(')')) continue;
    const options = line.slice(open + 2, -1).split(',').map((o) => o.trim());
    mounts.push({
      source: line.slice(0, on),
      mountPoint: line.slice(on + 4, open),
      fsType: options[0] || '',
      readOnly: options.includes('read-only') || options.includes('rdonly'),
      hidden: options.includes('nobrowse'),
    });
  }
  return mounts;
}

/** External disks, SD cards and disk images under /Volumes, and network shares (as Finder shows them). */
export function isEligibleMacMount(mount: MacMount): boolean {
  if (mount.hidden) return false;
  if (isNetworkFs(mount.fsType)) return !mount.mountPoint.startsWith('/System/');
  if (mount.fsType === 'devfs' || mount.fsType === 'autofs') return false;
  const match = /^\/Volumes\/([^/]+)$/.exec(mount.mountPoint);
  return Boolean(match) && !match![1].startsWith('.') && !match![1].startsWith('com.apple.');
}
