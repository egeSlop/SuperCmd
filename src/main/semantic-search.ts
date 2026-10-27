/**
 * semantic-search.ts
 *
 * Local semantic search over files, notes and clipboard history, powered by
 * Google's EmbeddingGemma 2 running in llama.cpp on Metal.
 *
 * - Embeddings come from the native `embedding-server` helper (see
 *   src/native/embedding-server.swift). It is spawned on demand and exits
 *   after a short idle period, so nothing stays resident when unused.
 * - Text uses the 270M text backbone (Unsloth dynamic Q4_K_XL by default).
 *   Images, video and audio additionally use the media encoder (mmproj),
 *   which the helper only loads while media is being indexed: queries are
 *   always text, and the model maps every modality into one vector space.
 * - Vectors are Matryoshka-truncated to 256 dims and stored as int8 with a
 *   per-vector scale (~260 bytes per chunk); search is a brute-force scan.
 * - Indexing is incremental (mtime/size, then a content hash for text) and by
 *   default only runs while the Mac is on power; searching always works.
 *   The index is unloaded from memory after a few idle minutes.
 *
 * Config and index are machine-local (models and folders don't belong in the
 * synced settings file), stored under <userData>/semantic-search/.
 *
 * External and network drives the user opts in carry their own index on the
 * drive, in a format shared with krunner-embed on Linux
 * (docs/semantic-index-format.md, semantic-search-drives.ts); searches cover
 * the home index and every connected drive.
 */

import { app, powerMonitor } from 'electron';
import * as fs from 'fs';
import * as path from 'path';
import * as os from 'os';
import { createHash } from 'crypto';
import { spawn, type ChildProcess } from 'child_process';
import { getNativeBinaryPath } from './native-binary';
import { getAllNotes } from './notes-store';
import { getClipboardHistory } from './clipboard-manager';
import {
  CHUNK_CHARS,
  DRIVE_KINDS,
  MAX_CHUNKS_PER_ITEM,
  QUERY_CENTROID_PROBES,
  QUERY_DEBIAS_STRENGTH,
  bestRowScore,
  chunkText,
  debiasQuery,
  driveLockOwner,
  meanDirection,
  quantizeRows,
  queryTerms,
  sameMtime,
  standoutScores,
  titleBoost,
  type DriveIndexItem,
  type DriveItemKind,
  type DriveLockInfo,
} from './semantic-search-core';
import {
  DriveLock,
  breakDriveLock,
  driveIndexBytes,
  driveIndexStamp,
  isVolumeMounted,
  isVolumeWritable,
  listVolumes,
  readDriveIndex,
  setVolumeEnabled,
  startDriveMonitor,
  stopDriveMonitor,
  writeDriveIndex,
  type DriveVolume,
} from './semantic-search-drives';

// ─── Models ─────────────────────────────────────────────────────────

export type SemanticModelVariant = 'compact' | 'balanced' | 'precise';

type ModelFile = { fileName: string; url: string; sha256: string; bytes: number };

// Revisions are pinned so the SHA-256 checks below stay meaningful.
const UNSLOTH_REPO = 'https://huggingface.co/unsloth/embeddinggemma-2-GGUF/resolve/031f0d4b35536f69ab3509d4893c923264fcf253';
const GGML_ORG_REPO = 'https://huggingface.co/ggml-org/embeddinggemma-2-GGUF/resolve/58faf90940886df2d62be555c8765d00dac362c5';

// Measured on an M1 Pro against the BF16 reference (mean cosine over 465
// texts): compact 0.994, balanced 0.9985, precise 0.9999. Query latency is
// ~20ms on Metal for all three; the difference is size and RAM.
const TEXT_MODELS: Record<SemanticModelVariant, ModelFile> = {
  compact: {
    fileName: 'embeddinggemma-2-UD-Q4_K_XL.gguf',
    url: `${UNSLOTH_REPO}/embeddinggemma-2-UD-Q4_K_XL.gguf`,
    sha256: 'ea905fd08e8061db77a0031cbb5096d459d7cfd2bdd0ad2ba66e483f16719493',
    bytes: 175_673_856,
  },
  balanced: {
    fileName: 'embeddinggemma-2-UD-Q5_K_XL.gguf',
    url: `${UNSLOTH_REPO}/embeddinggemma-2-UD-Q5_K_XL.gguf`,
    sha256: 'a9d7a3b7eb421113f05e964e10882676f10fff83811b7876d5bb983db7c7ac41',
    bytes: 210_055_680,
  },
  precise: {
    fileName: 'embeddinggemma-2-Q8_0.gguf',
    url: `${GGML_ORG_REPO}/embeddinggemma-2-Q8_0.gguf`,
    sha256: '2188ac1deca4b77dffefd603c2776a9d76d9d74ec01841392982ebb840b09135',
    bytes: 309_855_456,
  },
};

// Vision (170M) + audio (300M) encoders. Q8_0 is the smallest published
// quantization of the projector.
const MEDIA_MODEL: ModelFile = {
  fileName: 'mmproj-embeddinggemma-2-Q8_0.gguf',
  url: `${GGML_ORG_REPO}/mmproj-embeddinggemma-2-Q8_0.gguf`,
  sha256: 'c4a8a52691ecef40618438928bdf9e68379b854e24166f292592353db0aab64f',
  bytes: 554_821_024,
};

const EMBEDDING_DIM = 256;
const INDEX_VERSION = 1;
const QUERY_PREFIX = 'task: search result | query: ';
const HELPER_IDLE_EXIT_MS = 2 * 60 * 1000;
const HELPER_REQUEST_TIMEOUT_MS = 2 * 60 * 1000;
const BACKGROUND_REFRESH_MS = 30 * 60 * 1000;
const FIRST_REFRESH_DELAY_MS = 45 * 1000;
const STALE_INDEX_MS = 5 * 60 * 1000;
const POWER_CONNECTED_DELAY_MS = 15 * 1000;
const INDEX_IDLE_UNLOAD_MS = 5 * 60 * 1000;

const TEXT_BATCH_SIZE = 8;
const MAX_TEXT_FILE_BYTES = 2 * 1024 * 1024;
const MAX_RICH_FILE_BYTES = 60 * 1024 * 1024;
const MAX_IMAGE_BYTES = 40 * 1024 * 1024;
const MIN_IMAGE_BYTES = 12 * 1024;
const MAX_AUDIO_BYTES = 300 * 1024 * 1024;
const MAX_FILES = 20_000;
const MAX_WALK_DEPTH = 8;
const MAX_CLIPBOARD_ITEMS = 500;

const PLAIN_TEXT_EXTENSIONS = new Set([
  'txt', 'md', 'markdown', 'mdx', 'rst', 'org', 'tex', 'csv', 'tsv', 'log', 'json', 'yaml', 'yml', 'toml', 'ini',
  'xml', 'sql', 'sh', 'zsh', 'bash', 'fish', 'js', 'jsx', 'mjs', 'cjs', 'ts', 'tsx', 'py', 'rb', 'go', 'rs', 'swift',
  'kt', 'kts', 'java', 'scala', 'c', 'h', 'cc', 'cpp', 'hpp', 'm', 'mm', 'cs', 'php', 'lua', 'r', 'dart', 'ex', 'exs',
  'erl', 'hs', 'clj', 'vue', 'svelte', 'css', 'scss', 'less', 'graphql', 'proto', 'ipynb',
]);
const RICH_TEXT_EXTENSIONS = new Set(['pdf', 'rtf', 'doc', 'docx', 'odt', 'html', 'htm', 'webarchive']);
const IMAGE_EXTENSIONS = new Set(['jpg', 'jpeg', 'png', 'heic', 'heif', 'gif', 'webp', 'tif', 'tiff', 'bmp']);
const VIDEO_EXTENSIONS = new Set(['mp4', 'mov', 'm4v']);
const AUDIO_EXTENSIONS = new Set(['mp3', 'm4a', 'wav', 'aif', 'aiff', 'flac', 'aac', 'caf']);
const SKIPPED_DIRECTORY_NAMES = new Set([
  'node_modules', 'Library', 'build', 'dist', 'out', 'target', 'vendor', 'Pods', 'DerivedData', 'venv', 'env',
  '__pycache__', 'coverage', 'bower_components', 'site-packages',
]);
// macOS packages look like folders but are opaque documents/apps.
const PACKAGE_DIRECTORY_EXTENSIONS = new Set([
  'app', 'photoslibrary', 'photolibrary', 'musiclibrary', 'tvlibrary', 'imovielibrary', 'fcpbundle', 'logicx', 'band',
  'bundle', 'framework', 'plugin', 'kext', 'xcarchive', 'xcassets', 'xcodeproj', 'xcworkspace', 'pages', 'numbers',
  'key', 'rtfd', 'sparsebundle', 'pkg', 'lproj',
]);

// ─── Types ──────────────────────────────────────────────────────────

export type SemanticItemKind = 'document' | 'image' | 'video' | 'audio' | 'note' | 'clipboard';

export interface SemanticSearchConfig {
  enabled: boolean;
  modelVariant: SemanticModelVariant;
  includeMedia: boolean;
  includeNotes: boolean;
  includeClipboard: boolean;
  folders: string[];
  /** Background indexing waits for AC power (manual "Update Now" still runs). */
  onlyWhileCharging: boolean;
  /** Files that match by meaning get a section in the main launcher search. */
  showInLauncher: boolean;
}

export interface SemanticModelFileStatus {
  state: 'not-downloaded' | 'downloading' | 'downloaded' | 'error';
  fileName: string;
  bytesDownloaded: number;
  totalBytes: number;
  error?: string;
}

export interface SemanticDriveStatus {
  /** Mount point */
  root: string;
  label: string;
  fsType: string;
  network: boolean;
  readOnly: boolean;
  bytesTotal: number;
  /** Opted in (on this drive, or for read-only drives on this Mac). */
  enabled: boolean;
  storage?: 'drive' | 'local';
  writable?: boolean;
  items?: number;
  chunks?: number;
  diskBytes?: number;
  lastIndexedAt?: number | null;
  writer?: string;
  incompatible?: boolean;
  state?: 'idle' | 'indexing' | 'loading' | 'locked';
  /** Another machine is updating the index. */
  lock?: { owner: string; heartbeat: number | null };
  error?: string;
}

export interface SemanticSearchStatus {
  config: SemanticSearchConfig;
  volumes: SemanticDriveStatus[];
  helperAvailable: boolean;
  textModel: SemanticModelFileStatus;
  mediaModel: SemanticModelFileStatus;
  ready: boolean;
  index: {
    state: 'idle' | 'indexing' | 'error';
    processed: number;
    total: number;
    /** Label of the drive being indexed; unset for this Mac's index. */
    target?: string;
    items: number;
    chunks: number;
    lastIndexedAt: number | null;
    error?: string;
    /** Background indexing is held back because the Mac is on battery. */
    pausedOnBattery: boolean;
  };
}

export interface SemanticSearchResult {
  id: string;
  kind: SemanticItemKind;
  title: string;
  subtitle: string;
  preview: string;
  score: number;
  /** How far the score stands out from every other candidate (leave-one-out z-score). */
  z: number;
  path?: string;
  noteId?: string;
  clipboardId?: string;
  updatedAt: number;
}

interface IndexedItem {
  key: string;
  kind: SemanticItemKind;
  title: string;
  subtitle: string;
  path?: string;
  noteId?: string;
  clipboardId?: string;
  /** mtime / updatedAt / timestamp — changes trigger re-embedding. */
  version: number;
  size: number;
  updatedAt: number;
  /** Hash of the embedded text, so touched-but-unchanged files aren't re-embedded. */
  contentHash?: string;
  previews: string[];
  /** int8 vectors, `previews.length || 1` rows of EMBEDDING_DIM. */
  vectors: Int8Array;
  /** Per-row dequantization scale. */
  scales: Float32Array;
}

interface IndexMetaFile {
  version: number;
  modelVariant: SemanticModelVariant;
  dim: number;
  lastIndexedAt: number | null;
  items: Array<Omit<IndexedItem, 'vectors' | 'scales'> & { rows: number }>;
}

interface Candidate {
  key: string;
  kind: SemanticItemKind;
  title: string;
  subtitle: string;
  path?: string;
  noteId?: string;
  clipboardId?: string;
  version: number;
  size: number;
  updatedAt: number;
  text?: string;
}

// ─── Paths & config ─────────────────────────────────────────────────

function getRootDir(): string {
  return path.join(app.getPath('userData'), 'semantic-search');
}

function getModelsDir(): string {
  return path.join(getRootDir(), 'models');
}

function getConfigPath(): string {
  return path.join(getRootDir(), 'config.json');
}

function getIndexMetaPath(): string {
  return path.join(getRootDir(), 'index.json');
}

function getIndexVectorsPath(): string {
  return path.join(getRootDir(), 'index.bin');
}

function defaultFolders(): string[] {
  const home = os.homedir();
  return ['Documents', 'Desktop', 'Downloads'].map((name) => path.join(home, name));
}

const DEFAULT_CONFIG: SemanticSearchConfig = {
  enabled: false,
  modelVariant: 'compact',
  includeMedia: true,
  includeNotes: true,
  includeClipboard: true,
  folders: [],
  onlyWhileCharging: true,
  showInLauncher: true,
};

let configCache: SemanticSearchConfig | null = null;

function normalizeConfig(raw: any): SemanticSearchConfig {
  const variant = raw?.modelVariant;
  const folders = Array.isArray(raw?.folders)
    ? Array.from(new Set(raw.folders.filter((f: unknown) => typeof f === 'string' && path.isAbsolute(f)).map((f: string) => path.normalize(f))))
    : defaultFolders();
  return {
    enabled: typeof raw?.enabled === 'boolean' ? raw.enabled : DEFAULT_CONFIG.enabled,
    modelVariant: variant === 'balanced' || variant === 'precise' ? variant : 'compact',
    includeMedia: typeof raw?.includeMedia === 'boolean' ? raw.includeMedia : DEFAULT_CONFIG.includeMedia,
    includeNotes: typeof raw?.includeNotes === 'boolean' ? raw.includeNotes : DEFAULT_CONFIG.includeNotes,
    includeClipboard: typeof raw?.includeClipboard === 'boolean' ? raw.includeClipboard : DEFAULT_CONFIG.includeClipboard,
    folders: folders as string[],
    onlyWhileCharging: typeof raw?.onlyWhileCharging === 'boolean' ? raw.onlyWhileCharging : DEFAULT_CONFIG.onlyWhileCharging,
    showInLauncher: typeof raw?.showInLauncher === 'boolean' ? raw.showInLauncher : DEFAULT_CONFIG.showInLauncher,
  };
}

export function getSemanticSearchConfig(): SemanticSearchConfig {
  if (configCache) return { ...configCache, folders: [...configCache.folders] };
  let raw: any = null;
  try {
    raw = JSON.parse(fs.readFileSync(getConfigPath(), 'utf8'));
  } catch {}
  configCache = normalizeConfig(raw);
  return { ...configCache, folders: [...configCache.folders] };
}

function writeFileAtomic(filePath: string, data: string | Buffer): void {
  fs.mkdirSync(path.dirname(filePath), { recursive: true });
  const tempPath = `${filePath}.tmp`;
  fs.writeFileSync(tempPath, data);
  fs.renameSync(tempPath, filePath);
}

export function updateSemanticSearchConfig(patch: Partial<SemanticSearchConfig>): SemanticSearchStatus {
  const previous = getSemanticSearchConfig();
  const next = normalizeConfig({ ...previous, ...patch });
  configCache = next;
  writeFileAtomic(getConfigPath(), JSON.stringify(next, null, 2));

  if (next.modelVariant !== previous.modelVariant) {
    // Vectors from another quantization are close but not identical; rebuild.
    cancelIndexing();
    stopHelper();
    clearIndex();
    removeUnusedTextModels(next.modelVariant);
  }
  if (next.includeMedia !== previous.includeMedia) {
    stopHelper();
  }
  if (!next.enabled) {
    cancelIndexing();
    stopHelper();
  }
  if (next.enabled !== previous.enabled) syncDriveMonitor(next);
  if (!next.onlyWhileCharging && previous.onlyWhileCharging && indexingDeferred) {
    scheduleIndexing('charging-requirement-off', 1500);
  }

  if (next.enabled) {
    const needsModels = !previous.enabled
      || next.modelVariant !== previous.modelVariant
      || (next.includeMedia && !previous.includeMedia);
    const sourcesChanged = next.includeMedia !== previous.includeMedia
      || next.includeNotes !== previous.includeNotes
      || next.includeClipboard !== previous.includeClipboard
      || next.folders.join('\n') !== previous.folders.join('\n');
    // The download kicks off indexing itself once the files are in place.
    if (needsModels) void downloadSemanticSearchModels();
    else if (sourcesChanged) scheduleIndexing('config-changed', 1500);
  }
  return getSemanticSearchStatus();
}

// ─── Model downloads ────────────────────────────────────────────────

const downloadStatus = new Map<string, SemanticModelFileStatus>();
const downloadPromises = new Map<string, Promise<void>>();

function modelPath(file: ModelFile): string {
  return path.join(getModelsDir(), file.fileName);
}

function getModelFileStatus(file: ModelFile): SemanticModelFileStatus {
  const active = downloadStatus.get(file.fileName);
  if (active && active.state !== 'downloaded') return { ...active };
  try {
    if (fs.statSync(modelPath(file)).size === file.bytes) {
      return { state: 'downloaded', fileName: file.fileName, bytesDownloaded: file.bytes, totalBytes: file.bytes };
    }
  } catch {}
  return { state: 'not-downloaded', fileName: file.fileName, bytesDownloaded: 0, totalBytes: file.bytes };
}

function isModelReady(file: ModelFile): boolean {
  return getModelFileStatus(file).state === 'downloaded';
}

function httpGetFollowingRedirects(url: string, headers: Record<string, string>, redirectsLeft = 5): Promise<any> {
  return new Promise((resolve, reject) => {
    const request = require('https').get(url, { headers: { 'User-Agent': 'SuperCmd/1.0 semantic-search', Accept: '*/*', ...headers } }, (response: any) => {
      const status = Number(response.statusCode || 0);
      const location = String(response.headers?.location || '');
      if (status >= 300 && status < 400 && location) {
        response.resume();
        if (redirectsLeft <= 0) {
          reject(new Error('Too many redirects'));
          return;
        }
        resolve(httpGetFollowingRedirects(new URL(location, url).toString(), headers, redirectsLeft - 1));
        return;
      }
      if (status >= 400) {
        response.resume();
        reject(new Error(`HTTP ${status} while downloading ${path.basename(new URL(url).pathname)}`));
        return;
      }
      resolve(response);
    });
    request.on('error', reject);
    request.setTimeout(60_000, () => request.destroy(new Error('Download timed out')));
  });
}

async function sha256OfFile(filePath: string): Promise<string> {
  const hash = createHash('sha256');
  await new Promise<void>((resolve, reject) => {
    fs.createReadStream(filePath)
      .on('data', (chunk) => hash.update(chunk))
      .on('end', () => resolve())
      .on('error', reject);
  });
  return hash.digest('hex');
}

/** Downloads with resume support (HTTP Range) and SHA-256 verification. */
function downloadModelFile(file: ModelFile): Promise<void> {
  if (isModelReady(file)) return Promise.resolve();
  const existing = downloadPromises.get(file.fileName);
  if (existing) return existing;

  const finalPath = modelPath(file);
  const partialPath = `${finalPath}.download`;
  const setStatus = (status: Partial<SemanticModelFileStatus>) => {
    const prev = downloadStatus.get(file.fileName);
    downloadStatus.set(file.fileName, {
      state: 'downloading',
      fileName: file.fileName,
      bytesDownloaded: prev?.bytesDownloaded ?? 0,
      totalBytes: file.bytes,
      ...status,
    });
  };

  const promise = (async () => {
    fs.mkdirSync(path.dirname(finalPath), { recursive: true });
    let offset = 0;
    try {
      offset = fs.statSync(partialPath).size;
    } catch {}
    if (offset > file.bytes) {
      fs.rmSync(partialPath, { force: true });
      offset = 0;
    }
    setStatus({ state: 'downloading', bytesDownloaded: offset, error: undefined });

    if (offset < file.bytes) {
      const response = await httpGetFollowingRedirects(file.url, offset > 0 ? { Range: `bytes=${offset}-` } : {});
      const resumed = Number(response.statusCode) === 206;
      if (!resumed) offset = 0;
      await new Promise<void>((resolve, reject) => {
        const out = fs.createWriteStream(partialPath, { flags: resumed ? 'a' : 'w' });
        let received = offset;
        response.on('data', (chunk: Buffer) => {
          received += chunk.length;
          setStatus({ bytesDownloaded: received });
        });
        response.on('error', reject);
        out.on('error', reject);
        out.on('finish', () => resolve());
        response.pipe(out);
      });
    }

    // A dropped connection leaves a short file: keep it so the next attempt resumes.
    if (fs.statSync(partialPath).size !== file.bytes) {
      throw new Error(`Download of ${file.fileName} was interrupted`);
    }
    const digest = await sha256OfFile(partialPath);
    if (digest !== file.sha256) {
      fs.rmSync(partialPath, { force: true });
      throw new Error(`Integrity check failed for ${file.fileName}`);
    }
    fs.renameSync(partialPath, finalPath);
    downloadStatus.set(file.fileName, { state: 'downloaded', fileName: file.fileName, bytesDownloaded: file.bytes, totalBytes: file.bytes });
    console.log(`[SemanticSearch] Downloaded ${file.fileName}`);
  })()
    .catch((error) => {
      const message = error instanceof Error ? error.message : String(error);
      console.error(`[SemanticSearch] Download failed for ${file.fileName}:`, message);
      setStatus({ state: 'error', error: message });
      throw error;
    })
    .finally(() => {
      downloadPromises.delete(file.fileName);
    });

  downloadPromises.set(file.fileName, promise);
  return promise;
}

/** Starts downloading whatever the current config needs; resolves when done. */
export async function downloadSemanticSearchModels(): Promise<SemanticSearchStatus> {
  const config = getSemanticSearchConfig();
  const jobs = [downloadModelFile(TEXT_MODELS[config.modelVariant])];
  if (config.includeMedia) jobs.push(downloadModelFile(MEDIA_MODEL));
  const results = await Promise.allSettled(jobs);
  if (results.every((r) => r.status === 'fulfilled') && getSemanticSearchConfig().enabled) {
    scheduleIndexing('models-downloaded', 500);
  }
  return getSemanticSearchStatus();
}

function removeUnusedTextModels(keep: SemanticModelVariant): void {
  for (const [variant, file] of Object.entries(TEXT_MODELS)) {
    if (variant === keep) continue;
    fs.rmSync(modelPath(file), { force: true });
    fs.rmSync(`${modelPath(file)}.download`, { force: true });
  }
}

// ─── Helper process ─────────────────────────────────────────────────

/** The helper itself is unusable (missing, crashed, stopped) — not a per-item failure. */
class HelperUnavailableError extends Error {}

type HelperRequest = {
  payload: Record<string, unknown>;
  resolve: (value: any) => void;
  reject: (error: Error) => void;
};

let helperProcess: ChildProcess | null = null;
let helperStarting: Promise<void> | null = null;
let helperBuffer = '';
let helperMediaEnabled = false;
let helperInFlight: (HelperRequest & { id: number; timer: NodeJS.Timeout }) | null = null;
let helperIdleTimer: NodeJS.Timeout | null = null;
let helperNextId = 1;
// Set after media indexing: freeing the encoder in-process leaves ~85MB of
// allocator/Metal caches behind, so a fresh text-only process is leaner.
let helperRetireWhenIdle = false;
// Queries jump ahead of indexing work so search stays snappy mid-index.
const helperQueue: { high: HelperRequest[]; low: HelperRequest[] } = { high: [], low: [] };

function getHelperBinaryPath(): string {
  return getNativeBinaryPath('embedding-server');
}

function isHelperAvailable(): boolean {
  try {
    return fs.existsSync(getHelperBinaryPath());
  } catch {
    return false;
  }
}

function stopHelper(): void {
  if (helperIdleTimer) {
    clearTimeout(helperIdleTimer);
    helperIdleTimer = null;
  }
  const child = helperProcess;
  helperProcess = null;
  helperStarting = null;
  helperBuffer = '';
  helperRetireWhenIdle = false;
  if (child) {
    try {
      child.stdin?.write('{"command":"exit"}\n');
    } catch {}
    setTimeout(() => {
      try { child.kill(); } catch {}
    }, 1000);
  }
  const error = new HelperUnavailableError('Embedding helper stopped');
  if (helperInFlight) {
    clearTimeout(helperInFlight.timer);
    helperInFlight.reject(error);
    helperInFlight = null;
  }
  for (const req of [...helperQueue.high, ...helperQueue.low]) req.reject(error);
  helperQueue.high = [];
  helperQueue.low = [];
}

function armHelperIdleTimer(): void {
  if (helperIdleTimer) clearTimeout(helperIdleTimer);
  helperIdleTimer = setTimeout(() => {
    if (!helperInFlight && helperQueue.high.length === 0 && helperQueue.low.length === 0) {
      console.log('[SemanticSearch] Embedding helper idle, exiting');
      stopHelper();
    } else {
      armHelperIdleTimer();
    }
  }, HELPER_IDLE_EXIT_MS);
}

function ensureHelper(): Promise<void> {
  const config = getSemanticSearchConfig();
  const wantMedia = config.includeMedia && isModelReady(MEDIA_MODEL);
  if (helperProcess && helperMediaEnabled !== wantMedia) stopHelper();
  if (helperProcess && !helperStarting) return Promise.resolve();
  if (helperStarting) return helperStarting;

  const textModel = TEXT_MODELS[config.modelVariant];
  if (!isModelReady(textModel)) return Promise.reject(new HelperUnavailableError('Semantic search model is not downloaded'));
  const binaryPath = getHelperBinaryPath();
  if (!fs.existsSync(binaryPath)) {
    return Promise.reject(new HelperUnavailableError('Semantic search helper is missing. Run npm run build:native to build it.'));
  }

  const args = ['serve', '--model', modelPath(textModel)];
  if (wantMedia) args.push('--mmproj', modelPath(MEDIA_MODEL));
  const child = spawn(binaryPath, args, { stdio: ['pipe', 'pipe', 'pipe'] });
  helperProcess = child;
  helperMediaEnabled = wantMedia;
  helperBuffer = '';

  helperStarting = new Promise<void>((resolve, reject) => {
    const startupTimeout = setTimeout(() => {
      reject(new HelperUnavailableError('Embedding helper startup timed out'));
      stopHelper();
    }, 60_000);

    child.stdout!.on('data', (chunk: Buffer) => {
      helperBuffer += chunk.toString();
      const lines = helperBuffer.split('\n');
      helperBuffer = lines.pop() || '';
      for (const line of lines) {
        if (!line.trim()) continue;
        let json: any;
        try {
          json = JSON.parse(line);
        } catch {
          continue;
        }
        if (json.ready) {
          clearTimeout(startupTimeout);
          helperStarting = null;
          resolve();
          pumpHelperQueue();
          continue;
        }
        if (json.error && json.id === undefined && helperStarting) {
          clearTimeout(startupTimeout);
          reject(new HelperUnavailableError(json.error));
          stopHelper();
          continue;
        }
        const inFlight = helperInFlight;
        if (inFlight && json.id === inFlight.id) {
          clearTimeout(inFlight.timer);
          helperInFlight = null;
          if (json.error) inFlight.reject(new Error(json.error));
          else inFlight.resolve(json);
          pumpHelperQueue();
        }
      }
    });
    child.stderr!.on('data', (chunk: Buffer) => {
      const text = chunk.toString().trim();
      if (text) console.warn('[SemanticSearch][helper]', text);
    });
    child.on('exit', (code) => {
      clearTimeout(startupTimeout);
      if (helperProcess !== child) return;
      console.log(`[SemanticSearch] Embedding helper exited with code ${code}`);
      helperProcess = null;
      if (helperStarting) {
        helperStarting = null;
        reject(new HelperUnavailableError(`Embedding helper exited during startup (code ${code})`));
      }
      if (helperInFlight) {
        clearTimeout(helperInFlight.timer);
        helperInFlight.reject(new HelperUnavailableError('Embedding helper exited'));
        helperInFlight = null;
      }
    });
  });
  return helperStarting;
}

function pumpHelperQueue(): void {
  if (helperInFlight || helperStarting || !helperProcess) return;
  const next = helperQueue.high.shift() || helperQueue.low.shift();
  if (!next) {
    if (helperRetireWhenIdle) stopHelper();
    else armHelperIdleTimer();
    return;
  }
  const id = helperNextId++;
  // A wedged decode (e.g. a damaged video) must not stall the queue forever.
  const timer = setTimeout(() => {
    console.warn('[SemanticSearch] Embedding helper request timed out, restarting helper');
    stopHelper();
  }, HELPER_REQUEST_TIMEOUT_MS);
  helperInFlight = { ...next, id, timer };
  try {
    helperProcess.stdin!.write(`${JSON.stringify({ id, ...next.payload })}\n`);
  } catch (error) {
    clearTimeout(timer);
    helperInFlight = null;
    next.reject(new HelperUnavailableError(error instanceof Error ? error.message : String(error)));
    pumpHelperQueue();
  }
}

async function callHelper(payload: Record<string, unknown>, priority: 'high' | 'low'): Promise<any> {
  await ensureHelper();
  return await new Promise((resolve, reject) => {
    helperQueue[priority].push({ payload, resolve, reject });
    pumpHelperQueue();
  });
}

function decodeFloat32(base64: string): Float32Array {
  const buffer = Buffer.from(base64, 'base64');
  const copy = new ArrayBuffer(buffer.byteLength);
  new Uint8Array(copy).set(buffer);
  return new Float32Array(copy);
}

async function embedTexts(texts: string[], priority: 'high' | 'low'): Promise<Float32Array[]> {
  const response = await callHelper({ command: 'embed', texts, dim: EMBEDDING_DIM }, priority);
  const flat = decodeFloat32(response.embeddings);
  const dim = Number(response.dim) || EMBEDDING_DIM;
  return Array.from({ length: Number(response.count) || 0 }, (_, i) => flat.subarray(i * dim, (i + 1) * dim));
}

// ─── Query correction ───────────────────────────────────────────────

let queryCentroid: { key: string; vector: Float32Array } | null = null;
let queryCentroidPending: { key: string; promise: Promise<Float32Array | null> } | null = null;

function getQueryCentroidPath(): string {
  return path.join(getRootDir(), 'query-centroid.json');
}

/**
 * The "average query" direction for the current model (see debiasQuery):
 * the mean of the probe queries' embeddings, cached on disk. Null if it
 * couldn't be computed; searches then run uncorrected and retry next time.
 */
function getQueryCentroid(): Promise<Float32Array | null> {
  const model = TEXT_MODELS[getSemanticSearchConfig().modelVariant];
  const key = createHash('sha1')
    .update(JSON.stringify([model.sha256, EMBEDDING_DIM, QUERY_PREFIX, QUERY_CENTROID_PROBES]))
    .digest('hex');
  if (queryCentroid?.key === key) return Promise.resolve(queryCentroid.vector);
  if (queryCentroidPending?.key === key) return queryCentroidPending.promise;

  const compute = async (): Promise<Float32Array | null> => {
    try {
      const stored = JSON.parse(fs.readFileSync(getQueryCentroidPath(), 'utf8'));
      if (stored?.key === key && Array.isArray(stored.vector) && stored.vector.length === EMBEDDING_DIM) {
        return Float32Array.from(stored.vector);
      }
    } catch {}
    const vectors = await embedTexts(QUERY_CENTROID_PROBES.map((probe) => `${QUERY_PREFIX}${probe}`), 'high');
    if (vectors.length !== QUERY_CENTROID_PROBES.length) return null;
    const vector = meanDirection(vectors, EMBEDDING_DIM);
    try {
      writeFileAtomic(getQueryCentroidPath(), JSON.stringify({ key, vector: Array.from(vector) }));
    } catch {}
    return vector;
  };
  const promise = compute().then(
    (vector) => {
      if (vector) queryCentroid = { key, vector };
      return vector;
    },
    (error) => {
      console.warn('[SemanticSearch] Query correction unavailable:', error instanceof Error ? error.message : error);
      return null;
    }
  ).finally(() => {
    if (queryCentroidPending?.promise === promise) queryCentroidPending = null;
  });
  queryCentroidPending = { key, promise };
  return promise;
}

// ─── Vector index ───────────────────────────────────────────────────

let indexItems: Map<string, IndexedItem> | null = null;
let indexLastIndexedAt: number | null = null;
let indexDirty = false;
let indexCounts: { items: number; chunks: number } | null = null;
let indexUnloadTimer: NodeJS.Timeout | null = null;

function countIndex(items: Map<string, IndexedItem>): { items: number; chunks: number } {
  let chunks = 0;
  let searchable = 0;
  for (const item of items.values()) {
    chunks += item.scales.length;
    if (item.scales.length > 0) searchable += 1;
  }
  return { items: searchable, chunks };
}

/** Drops the in-memory index after a few idle minutes; it reloads on demand. */
function armIndexUnloadTimer(): void {
  if (indexUnloadTimer) clearTimeout(indexUnloadTimer);
  indexUnloadTimer = setTimeout(() => {
    indexUnloadTimer = null;
    if (indexingPromise) {
      armIndexUnloadTimer();
      return;
    }
    try { saveIndex(); } catch {}
    if (indexItems) indexCounts = countIndex(indexItems);
    indexItems = null;
    for (const state of drives.values()) {
      if (!state.items || state.loading) continue;
      state.counts = countIndex(state.items);
      state.items = null;
    }
  }, INDEX_IDLE_UNLOAD_MS);
}

function loadIndex(): Map<string, IndexedItem> {
  armIndexUnloadTimer();
  if (indexItems) return indexItems;
  indexItems = new Map();
  try {
    const meta = JSON.parse(fs.readFileSync(getIndexMetaPath(), 'utf8')) as IndexMetaFile;
    const config = getSemanticSearchConfig();
    if (meta.version !== INDEX_VERSION || meta.dim !== EMBEDDING_DIM || meta.modelVariant !== config.modelVariant) {
      return indexItems;
    }
    const blob = fs.readFileSync(getIndexVectorsPath());
    let offset = 0;
    for (const item of meta.items) {
      const rows = item.rows;
      const vectorBytes = rows * EMBEDDING_DIM;
      const scaleBytes = rows * 4;
      if (offset + vectorBytes + scaleBytes > blob.byteLength) break;
      const vectors = new Int8Array(vectorBytes);
      vectors.set(new Int8Array(blob.buffer, blob.byteOffset + offset, vectorBytes));
      offset += vectorBytes;
      const scales = new Float32Array(rows);
      new Uint8Array(scales.buffer).set(blob.subarray(offset, offset + scaleBytes));
      offset += scaleBytes;
      const { rows: _rows, ...rest } = item;
      indexItems.set(item.key, { ...rest, vectors, scales });
    }
    indexLastIndexedAt = meta.lastIndexedAt ?? null;
  } catch {}
  indexCounts = countIndex(indexItems);
  return indexItems;
}

function saveIndex(): void {
  if (!indexItems || !indexDirty) return;
  const items = [...indexItems.values()];
  const buffers: Buffer[] = [];
  const metaItems: IndexMetaFile['items'] = [];
  for (const { vectors, scales, ...rest } of items) {
    buffers.push(Buffer.from(vectors.buffer, vectors.byteOffset, vectors.byteLength));
    buffers.push(Buffer.from(scales.buffer, scales.byteOffset, scales.byteLength));
    metaItems.push({ ...rest, rows: scales.length });
  }
  const meta: IndexMetaFile = {
    version: INDEX_VERSION,
    modelVariant: getSemanticSearchConfig().modelVariant,
    dim: EMBEDDING_DIM,
    lastIndexedAt: indexLastIndexedAt,
    items: metaItems,
  };
  writeFileAtomic(getIndexVectorsPath(), Buffer.concat(buffers));
  writeFileAtomic(getIndexMetaPath(), JSON.stringify(meta));
  indexDirty = false;
  indexCounts = countIndex(indexItems);
}

function clearIndex(): void {
  indexItems = new Map();
  indexCounts = { items: 0, chunks: 0 };
  indexLastIndexedAt = null;
  indexDirty = false;
  fs.rmSync(getIndexMetaPath(), { force: true });
  fs.rmSync(getIndexVectorsPath(), { force: true });
}

// ─── Candidate collection ───────────────────────────────────────────

function extensionOf(filePath: string): string {
  return path.extname(filePath).slice(1).toLowerCase();
}

function kindForExtension(ext: string, includeMedia: boolean): SemanticItemKind | null {
  if (PLAIN_TEXT_EXTENSIONS.has(ext) || RICH_TEXT_EXTENSIONS.has(ext)) return 'document';
  if (!includeMedia) return null;
  if (IMAGE_EXTENSIONS.has(ext)) return 'image';
  if (VIDEO_EXTENSIONS.has(ext)) return 'video';
  if (AUDIO_EXTENSIONS.has(ext)) return 'audio';
  return null;
}

function withinSizeLimits(kind: SemanticItemKind, ext: string, size: number): boolean {
  if (size <= 0) return false;
  if (kind === 'document') return size <= (RICH_TEXT_EXTENSIONS.has(ext) ? MAX_RICH_FILE_BYTES : MAX_TEXT_FILE_BYTES);
  if (kind === 'image') return size >= MIN_IMAGE_BYTES && size <= MAX_IMAGE_BYTES;
  if (kind === 'audio') return size <= MAX_AUDIO_BYTES;
  return true;
}

function tildePath(filePath: string): string {
  const home = os.homedir();
  return filePath.startsWith(home) ? `~${filePath.slice(home.length)}` : filePath;
}

// Windows and filesystem bookkeeping on drives (not hidden by a dot).
const DRIVE_SKIPPED_DIRECTORY_NAMES = new Set(['System Volume Information', '$RECYCLE.BIN', 'RECYCLER', 'lost+found']);

async function collectFileCandidates(
  config: { folders: string[]; includeMedia: boolean },
  isCancelled: () => boolean,
  options: { drive?: boolean } = {}
): Promise<Candidate[]> {
  const out: Candidate[] = [];
  const seen = new Set<string>();
  const walk = async (dir: string, depth: number): Promise<void> => {
    if (depth > MAX_WALK_DEPTH || out.length >= MAX_FILES || isCancelled()) return;
    let entries: fs.Dirent[];
    try {
      entries = await fs.promises.readdir(dir, { withFileTypes: true });
    } catch {
      return;
    }
    for (const entry of entries) {
      if (out.length >= MAX_FILES || isCancelled()) return;
      const name = entry.name;
      if (name.startsWith('.')) continue;
      const fullPath = path.join(dir, name);
      if (entry.isDirectory()) {
        if (SKIPPED_DIRECTORY_NAMES.has(name) || PACKAGE_DIRECTORY_EXTENSIONS.has(extensionOf(name))) continue;
        if (options.drive && DRIVE_SKIPPED_DIRECTORY_NAMES.has(name)) continue;
        await walk(fullPath, depth + 1);
        continue;
      }
      if (!entry.isFile()) continue;
      const ext = extensionOf(name);
      const kind = kindForExtension(ext, config.includeMedia);
      if (!kind || seen.has(fullPath)) continue;
      let stat: fs.Stats;
      try {
        stat = await fs.promises.stat(fullPath);
      } catch {
        continue;
      }
      if (!withinSizeLimits(kind, ext, stat.size)) continue;
      seen.add(fullPath);
      out.push({
        key: options.drive ? driveItemKey(fullPath) : `file:${fullPath}`,
        kind,
        title: name,
        subtitle: tildePath(path.dirname(fullPath)),
        path: fullPath,
        version: Math.floor(stat.mtimeMs),
        size: stat.size,
        updatedAt: Math.floor(stat.mtimeMs),
      });
    }
  };
  for (const folder of config.folders) {
    await walk(folder, 0);
  }
  return out;
}

function collectNoteCandidates(): Candidate[] {
  try {
    return getAllNotes()
      .filter((note) => `${note.title}${note.content}`.trim())
      .map((note) => ({
        key: `note:${note.id}`,
        kind: 'note' as const,
        title: note.title || 'Untitled Note',
        subtitle: 'Notes',
        noteId: note.id,
        version: note.updatedAt,
        size: note.content.length,
        updatedAt: note.updatedAt,
        text: note.content,
      }));
  } catch {
    return [];
  }
}

function collectClipboardCandidates(config: SemanticSearchConfig): Candidate[] {
  try {
    const out: Candidate[] = [];
    for (const item of getClipboardHistory().slice(0, MAX_CLIPBOARD_ITEMS)) {
      if (item.type === 'text' || item.type === 'url') {
        const text = String(item.content || '').trim();
        if (text.length < 3) continue;
        out.push({
          key: `clip:${item.id}`,
          kind: 'clipboard',
          title: text.replace(/\s+/g, ' ').slice(0, 120),
          subtitle: item.source ? `Clipboard · ${item.source}` : 'Clipboard',
          clipboardId: item.id,
          version: item.timestamp,
          size: text.length,
          updatedAt: item.timestamp,
          text,
        });
      } else if (item.type === 'image' && config.includeMedia && item.content && fs.existsSync(item.content)) {
        out.push({
          key: `clip:${item.id}`,
          kind: 'clipboard',
          title: item.preview || 'Image',
          subtitle: item.source ? `Clipboard image · ${item.source}` : 'Clipboard image',
          clipboardId: item.id,
          path: item.content,
          version: item.timestamp,
          size: item.metadata?.size || 0,
          updatedAt: item.timestamp,
        });
      }
    }
    return out;
  } catch {
    return [];
  }
}

// ─── Text extraction ────────────────────────────────────────────────

async function readDocumentText(candidate: Candidate): Promise<string> {
  if (candidate.text !== undefined) return candidate.text;
  if (!candidate.path) return '';
  const ext = extensionOf(candidate.path);
  if (RICH_TEXT_EXTENSIONS.has(ext)) {
    const response = await callHelper({ command: 'extract_text', path: candidate.path, max_chars: CHUNK_CHARS * MAX_CHUNKS_PER_ITEM }, 'low');
    return String(response.text || '');
  }
  const handle = await fs.promises.open(candidate.path, 'r');
  try {
    const buffer = Buffer.alloc(Math.min(candidate.size, CHUNK_CHARS * MAX_CHUNKS_PER_ITEM * 2));
    const { bytesRead } = await handle.read(buffer, 0, buffer.length, 0);
    const slice = buffer.subarray(0, bytesRead);
    if (slice.includes(0)) return ''; // binary file with a text-like extension
    return slice.toString('utf8');
  } finally {
    await handle.close();
  }
}

function mediaKindFor(candidate: Candidate): 'image' | 'video' | 'audio' | null {
  if (candidate.kind === 'image' || candidate.kind === 'video' || candidate.kind === 'audio') return candidate.kind;
  if (candidate.kind === 'clipboard' && candidate.path && candidate.text === undefined) return 'image';
  return null;
}

// ─── Indexing ───────────────────────────────────────────────────────

const indexStatus: Omit<SemanticSearchStatus['index'], 'items' | 'chunks' | 'lastIndexedAt' | 'pausedOnBattery'> = {
  state: 'idle',
  processed: 0,
  total: 0,
};
let indexingPromise: Promise<void> | null = null;
let indexingGeneration = 0;
let activeIndexingGeneration = 0;
let activeIndexingManual = false;
// A background run was skipped or paused for battery; run it on AC power.
let indexingDeferred = false;
let refreshTimer: NodeJS.Timeout | null = null;
let scheduledIndexTimer: NodeJS.Timeout | null = null;

function cancelIndexing(): void {
  indexingGeneration += 1;
}

function isReadyToIndex(config: SemanticSearchConfig): boolean {
  return config.enabled && isHelperAvailable() && isModelReady(TEXT_MODELS[config.modelVariant]);
}

function isOnBatteryPower(): boolean {
  try {
    return powerMonitor.isOnBatteryPower();
  } catch {
    return false;
  }
}

function isPausedOnBattery(config: SemanticSearchConfig): boolean {
  return config.enabled && config.onlyWhileCharging && isOnBatteryPower();
}

function scheduleIndexing(reason: string, delayMs: number, options: { manual?: boolean } = {}): void {
  if (scheduledIndexTimer) clearTimeout(scheduledIndexTimer);
  scheduledIndexTimer = setTimeout(() => {
    scheduledIndexTimer = null;
    void runIndexing(reason, options);
  }, delayMs);
}

/** Cheap identity for media: size + first and last 64 KiB (same as krunner-embed). */
async function mediaFingerprint(filePath: string, size: number): Promise<string | undefined> {
  const part = 64 * 1024;
  let handle: fs.promises.FileHandle | undefined;
  try {
    handle = await fs.promises.open(filePath, 'r');
    const hash = createHash('sha1');
    const sizeBytes = Buffer.alloc(8);
    sizeBytes.writeBigInt64LE(BigInt(size));
    hash.update(sizeBytes);
    const head = Buffer.alloc(Math.min(part, size));
    const first = await handle.read(head, 0, head.length, 0);
    hash.update(head.subarray(0, first.bytesRead));
    if (size > part) {
      const start = Math.max(part, size - part);
      const tail = Buffer.alloc(Math.min(part, size - start));
      const last = await handle.read(tail, 0, tail.length, start);
      hash.update(tail.subarray(0, last.bytesRead));
    }
    return hash.digest('hex');
  } catch {
    return undefined;
  } finally {
    await handle?.close();
  }
}

/** Where embedded items go: the home index or one drive's index. */
interface EmbedTarget {
  items: Map<string, IndexedItem>;
  markDirty(): void;
  /** Called every 200 files to keep progress; a drive may throw DriveUnavailableError. */
  checkpoint(): void;
  /** Recognise media by its fingerprint before re-embedding (drives: timestamps shift). */
  fingerprintMedia: boolean;
}

/** Embeds new and changed candidates into `target`. */
async function embedPending(
  pending: Candidate[],
  target: EmbedTarget,
  isCancelled: () => boolean
): Promise<{ completed: boolean; unchangedContent: number; embeddedMedia: boolean }> {
  const { items } = target;
  const store = (candidate: Candidate, previews: string[], vectors: Float32Array[], contentHash?: string) => {
    const { text: _text, ...rest } = candidate;
    items.set(candidate.key, { ...rest, contentHash, previews, ...quantizeRows(vectors, EMBEDDING_DIM) });
    target.markDirty();
  };
  const keep = (candidate: Candidate, existing: IndexedItem) => {
    const { text: _text, ...meta } = candidate;
    items.set(candidate.key, { ...existing, ...meta });
    target.markDirty();
  };
  let unchangedContent = 0;

  // Text: pack chunks from several items into each embed request.
  type Job = { candidate: Candidate; chunks: string[]; contentHash: string };
  let batch: Job[] = [];
  let batchChunks = 0;
  const flushText = async () => {
    if (batch.length === 0) return;
    const jobs = batch;
    batch = [];
    batchChunks = 0;
    // Clipboard snippets have no real title (model card: use "none").
    const texts = jobs.flatMap(({ candidate, chunks }) => chunks.map((chunk) =>
      `title: ${candidate.kind === 'clipboard' ? 'none' : candidate.title} | text: ${chunk}`));
    const vectors = await embedTexts(texts, 'low');
    let row = 0;
    for (const { candidate, chunks, contentHash } of jobs) {
      store(candidate, chunks.map((c) => c.slice(0, 160)), vectors.slice(row, row + chunks.length), contentHash);
      row += chunks.length;
    }
    indexStatus.processed += jobs.length;
  };

  if (pending.length > 0) await ensureHelper();

  let sinceSave = 0;
  let embeddedMedia = false;
  for (const candidate of pending) {
    if (isCancelled()) return { completed: false, unchangedContent, embeddedMedia };
    try {
      const mediaKind = mediaKindFor(candidate);
      if (mediaKind) {
        let fingerprint: string | undefined;
        if (target.fingerprintMedia && candidate.path) {
          fingerprint = await mediaFingerprint(candidate.path, candidate.size);
          const existing = items.get(candidate.key);
          if (fingerprint && existing && existing.contentHash === fingerprint && existing.scales.length > 0) {
            // Same bytes, new timestamp (copied, touched, other time zone).
            keep(candidate, existing);
            unchangedContent += 1;
            indexStatus.processed += 1;
            continue;
          }
        }
        const response = await callHelper({ command: 'embed_media', kind: mediaKind, path: candidate.path, dim: EMBEDDING_DIM }, 'low');
        store(candidate, [], [decodeFloat32(response.embedding)], fingerprint);
        indexStatus.processed += 1;
        embeddedMedia = true;
      } else {
        const text = await readDocumentText(candidate);
        // Touched but identical (sync tools, checkouts): keep the vectors.
        const contentHash = createHash('sha1').update(candidate.title).update('\0').update(text).digest('hex');
        const existing = items.get(candidate.key);
        if (existing && existing.contentHash === contentHash && existing.scales.length > 0) {
          keep(candidate, existing);
          unchangedContent += 1;
          indexStatus.processed += 1;
          continue;
        }
        // Files with no extractable text are still findable by name.
        const chunks = chunkText(text);
        batch.push({ candidate, chunks: chunks.length > 0 ? chunks : [candidate.title], contentHash });
        batchChunks += Math.max(1, chunks.length);
        if (batchChunks >= TEXT_BATCH_SIZE) await flushText();
      }
    } catch (error) {
      if (error instanceof HelperUnavailableError) throw error;
      console.warn(`[SemanticSearch] Skipping ${candidate.path || candidate.key}:`, error instanceof Error ? error.message : error);
      // Keep a zero-row entry so an unreadable file isn't retried every run
      // (it is retried once it changes on disk).
      store(candidate, [], []);
      indexStatus.processed += 1;
    }
    if (++sinceSave >= 200) {
      sinceSave = 0;
      target.checkpoint();
    }
  }
  try {
    await flushText();
  } catch (error) {
    if (error instanceof HelperUnavailableError) throw error;
    console.warn('[SemanticSearch] Final batch failed:', error instanceof Error ? error.message : error);
  }
  return { completed: !isCancelled(), unchangedContent, embeddedMedia };
}

async function indexHome(config: SemanticSearchConfig, mediaReady: boolean, reason: string, isCancelled: () => boolean): Promise<boolean> {
  const started = Date.now();
  const items = loadIndex();
  Object.assign(indexStatus, { state: 'indexing', target: undefined, processed: 0, total: 0, error: undefined });

  // Opted-in drives keep their own index, even if a folder points there.
  const driveRoots = [...drives.keys()];
  const folders = config.folders.filter((folder) => !driveRoots.some((root) => folder === root || folder.startsWith(`${root}/`)));
  const candidates: Candidate[] = [
    ...(config.includeNotes ? collectNoteCandidates() : []),
    ...(config.includeClipboard ? collectClipboardCandidates(config) : []),
    ...(await collectFileCandidates({ folders, includeMedia: config.includeMedia }, isCancelled)),
  ];
  if (isCancelled()) return false;

  const liveKeys = new Set(candidates.map((c) => c.key));
  for (const key of items.keys()) {
    if (!liveKeys.has(key)) {
      items.delete(key);
      indexDirty = true;
    }
  }

  const pending = candidates.filter((candidate) => {
    const existing = items.get(candidate.key);
    if (existing && existing.version === candidate.version && existing.size === candidate.size) return false;
    // Media waits until the encoder is downloaded.
    return !mediaKindFor(candidate) || mediaReady;
  });
  indexStatus.total = pending.length;
  console.log(`[SemanticSearch] Indexing (${reason}): ${pending.length} new/changed of ${candidates.length}`);

  const outcome = await embedPending(pending, {
    items,
    markDirty: () => { indexDirty = true; },
    checkpoint: () => saveIndex(),
    fingerprintMedia: false,
  }, isCancelled);
  if (!outcome.completed) return outcome.embeddedMedia;

  indexLastIndexedAt = Date.now();
  indexDirty = true;
  saveIndex();
  console.log(
    `[SemanticSearch] Index ready: ${items.size} items in ${((Date.now() - started) / 1000).toFixed(1)}s`
    + (outcome.unchangedContent > 0 ? ` (${outcome.unchangedContent} touched but unchanged, not re-embedded)` : '')
  );
  return outcome.embeddedMedia;
}

async function runIndexing(reason: string, options: { manual?: boolean } = {}): Promise<void> {
  const manual = Boolean(options.manual);
  if (indexingPromise) {
    if (activeIndexingGeneration === indexingGeneration) return indexingPromise;
    // A cancelled run is still winding down; start fresh once it has.
    await indexingPromise;
    if (indexingPromise) return indexingPromise;
  }
  const config = getSemanticSearchConfig();
  if (!isReadyToIndex(config)) return;
  if (!manual && isPausedOnBattery(config)) {
    indexingDeferred = true;
    return;
  }
  indexingDeferred = false;

  const generation = ++indexingGeneration;
  activeIndexingGeneration = generation;
  activeIndexingManual = manual;
  const isCancelled = () => generation !== indexingGeneration;
  const mediaReady = config.includeMedia && isModelReady(MEDIA_MODEL);

  indexingPromise = (async () => {
    let embeddedMedia = await indexHome(config, mediaReady, reason, isCancelled);
    if (!isCancelled()) embeddedMedia = (await indexDrives(config, mediaReady, reason, isCancelled)) || embeddedMedia;
    if (embeddedMedia && helperProcess) {
      helperRetireWhenIdle = true;
      pumpHelperQueue();
    }
  })()
    .catch((error) => {
      const message = error instanceof Error ? error.message : String(error);
      console.error('[SemanticSearch] Indexing failed:', message);
      indexStatus.error = message;
      try { saveIndex(); } catch {}
    })
    .finally(() => {
      if (generation !== indexingGeneration) {
        console.log(`[SemanticSearch] Indexing stopped after ${indexStatus.processed} of ${indexStatus.total}; progress kept`);
      }
      // Keep whatever a paused/cancelled run embedded; the next run resumes from there.
      try { saveIndex(); } catch {}
      indexStatus.state = indexStatus.error ? 'error' : 'idle';
      indexStatus.target = undefined;
      indexingPromise = null;
    });

  return indexingPromise;
}

// ─── Drives ─────────────────────────────────────────────────────────

const DRIVE_CONNECTED_DELAY_MS = 20 * 1000;
const DRIVE_HEARTBEAT_MS = 60 * 1000;
const DRIVE_SAVE_INTERVAL_MS = 2 * 60 * 1000; // whole-index rewrites on a USB stick add up
const DRIVE_RECHECK_MS = 30 * 1000; // notice other machines' updates to a drive index
const MAX_DRIVE_LOAD_WAIT_MS = 3000; // a slow share must not stall a search
const QUANT_NAMES: Record<SemanticModelVariant, string> = { compact: 'UD-Q4_K_XL', balanced: 'UD-Q5_K_XL', precise: 'Q8_0' };

/** The drive went away or another machine took over its lock: stop without writing. */
class DriveUnavailableError extends Error {}

interface DriveState {
  volume: DriveVolume;
  /** null while not in memory */
  items: Map<string, IndexedItem> | null;
  loading: Promise<void> | null;
  loadGeneration: number;
  /** index.bin as loaded (mtime:size), to notice updates by other machines */
  stamp: string;
  lastCheck: number;
  counts: { items: number; chunks: number };
  lastIndexedAt: number;
  lastWriter: string;
  incompatible: boolean;
  foreignLock: DriveLockInfo | null;
  error?: string;
  /** Opted out or unplugged: never written again. */
  closing: boolean;
}

const drives = new Map<string, DriveState>();
let activeDriveRoot: string | null = null;
let activeDriveAbort: string | null = null;

function driveWriter(): string {
  return `SuperCmd ${app.getVersion()} on ${os.hostname()}`;
}

/** Drive items are matched in NFC: Linux and macOS may spell the same name differently. */
function driveItemKey(fullPath: string): string {
  return `file:${fullPath.normalize('NFC')}`;
}

function toIndexedItem(volume: DriveVolume, item: DriveIndexItem): IndexedItem {
  const fullPath = path.join(volume.root, ...item.path.split('/'));
  return {
    key: driveItemKey(fullPath),
    kind: item.kind,
    title: path.basename(fullPath),
    subtitle: tildePath(path.dirname(fullPath)),
    path: fullPath,
    version: item.mtime,
    size: item.size,
    updatedAt: item.mtime,
    contentHash: item.hash.length > 0 ? Buffer.from(item.hash).toString('hex') : undefined,
    previews: item.previews,
    vectors: item.vectors,
    scales: item.scales,
  };
}

function toDriveItem(volume: DriveVolume, item: IndexedItem): DriveIndexItem | null {
  if (!item.path || !(DRIVE_KINDS as readonly string[]).includes(item.kind)) return null;
  const relative = path.relative(volume.root, item.path);
  if (!relative || relative.startsWith('..') || path.isAbsolute(relative)) return null;
  return {
    path: relative.split(path.sep).join('/'),
    kind: item.kind as DriveItemKind,
    mtime: item.version,
    size: item.size,
    hash: item.contentHash ? Uint8Array.from(Buffer.from(item.contentHash, 'hex')) : new Uint8Array(0),
    scales: item.scales,
    vectors: item.vectors,
    previews: item.previews,
  };
}

function loadDrive(state: DriveState): Promise<void> {
  const generation = ++state.loadGeneration;
  const volume = state.volume;
  const promise = readDriveIndex(volume.indexDir).then((result) => {
    if (state.loadGeneration !== generation || state.closing) return;
    const items = new Map<string, IndexedItem>();
    if (result.ok) {
      for (const item of result.items) {
        const indexed = toIndexedItem(volume, item);
        items.set(indexed.key, indexed);
      }
      state.lastIndexedAt = result.header.lastIndexedAt;
      state.lastWriter = result.header.writer;
    } else if (!result.incompatible) {
      // Damaged: rebuilt by the next run.
      console.warn(`[SemanticSearch] Could not read the index of ${volume.root}: ${result.error}`);
    }
    state.incompatible = !result.ok && result.incompatible;
    state.items = items;
    state.stamp = result.stamp;
    state.lastCheck = Date.now();
    state.counts = countIndex(items);
  }).finally(() => {
    if (state.loading === promise) state.loading = null;
  });
  state.loading = promise;
  return promise;
}

/** Loads (or reloads, when another machine updated them) the drive indexes for a search. */
async function ensureDrivesLoaded(): Promise<void> {
  const now = Date.now();
  const waits: Promise<void>[] = [];
  for (const state of drives.values()) {
    if (state.closing) continue;
    if (state.loading) {
      waits.push(state.loading);
    } else if (!state.items) {
      waits.push(loadDrive(state));
    } else if (activeDriveRoot !== state.volume.root && now - state.lastCheck >= DRIVE_RECHECK_MS) {
      state.lastCheck = now;
      if (driveIndexStamp(state.volume.indexDir) !== state.stamp) waits.push(loadDrive(state));
    }
  }
  if (waits.length === 0) return;
  await Promise.race([Promise.allSettled(waits), new Promise((resolve) => setTimeout(resolve, MAX_DRIVE_LOAD_WAIT_MS))]);
}

function syncDrives(): void {
  const enabled = new Map(listVolumes().filter((v) => v.enabled).map((v) => [v.root, v]));
  for (const [root, state] of drives) {
    const volume = enabled.get(root);
    if (volume && volume.dev === state.volume.dev && volume.indexDir === state.volume.indexDir) {
      state.volume = volume;
      continue;
    }
    state.closing = true;
    if (activeDriveRoot === root) activeDriveAbort = volume ? 'drive opted out' : 'drive removed';
    drives.delete(root);
    console.log(`[SemanticSearch] Drive ${volume ? 'opted out' : 'removed'}: ${root}`);
  }
  for (const [root, volume] of enabled) {
    if (drives.has(root)) continue;
    const state: DriveState = {
      volume,
      items: null,
      loading: null,
      loadGeneration: 0,
      stamp: '',
      lastCheck: 0,
      counts: { items: 0, chunks: 0 },
      lastIndexedAt: 0,
      lastWriter: '',
      incompatible: false,
      foreignLock: null,
      closing: false,
    };
    drives.set(root, state);
    console.log(`[SemanticSearch] Drive connected: ${root} (index kept ${volume.local ? 'on this Mac' : 'on the drive'})`);
    armIndexUnloadTimer();
    void loadDrive(state).then(() => {
      // Bring it up to date after a moment, unless it was indexed recently.
      if (!isVolumeWritable(volume) || !isReadyToIndex(getSemanticSearchConfig())) return;
      if (Date.now() - state.lastIndexedAt > BACKGROUND_REFRESH_MS && !scheduledIndexTimer && !indexingPromise) {
        scheduleIndexing('drive-connected', DRIVE_CONNECTED_DELAY_MS);
      }
    });
  }
}

async function indexDrives(config: SemanticSearchConfig, mediaReady: boolean, reason: string, isCancelled: () => boolean): Promise<boolean> {
  let embeddedMedia = false;
  for (const state of [...drives.values()]) {
    if (isCancelled()) break;
    const volume = state.volume;
    if (state.closing || !isVolumeWritable(volume)) continue;
    const lock = new DriveLock(volume.indexDir, `SuperCmd ${app.getVersion()}`);
    const { result, holder } = lock.acquire();
    if (result === 'held') {
      // Someone else is updating it: their index is searched as it is, and
      // the next run tries again.
      state.foreignLock = holder ?? null;
      state.error = `Being updated by ${holder ? driveLockOwner(holder) : 'another machine'}`;
      console.log(`[SemanticSearch] ${volume.root}: ${state.error}; skipping it for now`);
      continue;
    }
    if (result === 'error') {
      state.error = `Could not lock the index: ${lock.error}`;
      continue;
    }
    state.foreignLock = null;
    state.error = undefined;
    activeDriveRoot = volume.root;
    activeDriveAbort = null;
    const heartbeat = setInterval(() => {
      if (!lock.heartbeat()) activeDriveAbort = 'another machine took over the update';
    }, DRIVE_HEARTBEAT_MS);
    const blocked = (): string | null => {
      if (activeDriveAbort) return activeDriveAbort;
      if (state.closing || !isVolumeMounted(volume.root, volume.dev)) return 'drive removed';
      if (!lock.verify()) return 'another machine took over the update';
      return null;
    };
    const driveCancelled = () => isCancelled() || Boolean(activeDriveAbort) || state.closing;
    Object.assign(indexStatus, { target: volume.label, processed: 0, total: 0 });
    try {
      // Read it again under the lock: another machine may have updated it.
      if (!state.items || state.loading || driveIndexStamp(volume.indexDir) !== state.stamp) await loadDrive(state);
      if (state.incompatible) {
        state.error = `Its index was made with a different model or format (${state.lastWriter || 'unknown app'})`;
        continue;
      }
      const items = state.items ?? new Map<string, IndexedItem>();
      state.items = items;
      let dirty = false;
      let lastSave = Date.now();
      const save = () => {
        const why = blocked();
        if (why) throw new DriveUnavailableError(why);
        const stored = [...items.values()].map((item) => toDriveItem(volume, item)).filter((item): item is DriveIndexItem => item !== null);
        state.stamp = writeDriveIndex(volume.indexDir, { quant: QUANT_NAMES[config.modelVariant], lastIndexedAt: state.lastIndexedAt, writer: driveWriter() }, stored);
        state.lastWriter = driveWriter();
        dirty = false;
        lastSave = Date.now();
      };

      const candidates = await collectFileCandidates({ folders: [volume.root], includeMedia: config.includeMedia }, driveCancelled, { drive: true });
      if (driveCancelled()) {
        if (blocked()) throw new DriveUnavailableError(blocked()!);
        continue;
      }
      // A drive unplugged during the walk looks empty: don't drop its index.
      const why = blocked();
      if (why) throw new DriveUnavailableError(why);

      const live = new Set(candidates.map((c) => c.key));
      for (const key of [...items.keys()]) {
        if (!live.has(key)) {
          items.delete(key);
          dirty = true;
        }
      }
      const pending: Candidate[] = [];
      for (const candidate of candidates) {
        const existing = items.get(candidate.key);
        // FAT/exFAT and shares round timestamps and shift them with the time zone.
        if (existing && existing.size === candidate.size && sameMtime(existing.version, candidate.version, true)) {
          if (existing.path !== candidate.path) {
            // The name as found here (Linux may have stored another Unicode form).
            items.set(candidate.key, { ...existing, path: candidate.path, title: candidate.title, subtitle: candidate.subtitle });
            dirty = true;
          }
          continue;
        }
        if (!mediaKindFor(candidate) || mediaReady) pending.push(candidate);
      }
      indexStatus.total = pending.length;
      console.log(`[SemanticSearch] Indexing ${volume.root} (${reason}): ${pending.length} new/changed of ${candidates.length}`);

      const outcome = await embedPending(pending, {
        items,
        markDirty: () => { dirty = true; },
        checkpoint: () => {
          if (dirty && Date.now() - lastSave >= DRIVE_SAVE_INTERVAL_MS) save();
        },
        fingerprintMedia: true,
      }, driveCancelled);
      embeddedMedia = embeddedMedia || outcome.embeddedMedia;
      if (outcome.completed) state.lastIndexedAt = Date.now();
      if (dirty || outcome.completed) save();
      state.counts = countIndex(items);
    } catch (error) {
      if (!(error instanceof DriveUnavailableError)) throw error;
      state.error = error.message;
      // Not written: what is on the drive is the truth.
      state.items = null;
      console.log(`[SemanticSearch] Stopped indexing ${volume.root}: ${error.message}`);
    } finally {
      clearInterval(heartbeat);
      lock.release();
      activeDriveRoot = null;
      activeDriveAbort = null;
    }
  }
  return embeddedMedia;
}

function driveStatus(volume: DriveVolume): SemanticDriveStatus {
  const status: SemanticDriveStatus = {
    root: volume.root,
    label: volume.label,
    fsType: volume.fsType,
    network: volume.network,
    readOnly: volume.readOnly,
    bytesTotal: volume.bytesTotal,
    enabled: volume.enabled,
  };
  const state = volume.enabled ? drives.get(volume.root) : undefined;
  if (!state) return status;
  const counts = state.items ? countIndex(state.items) : state.counts;
  let driveState: SemanticDriveStatus['state'] = 'idle';
  if (activeDriveRoot === volume.root) driveState = 'indexing';
  else if (state.loading) driveState = 'loading';
  else if (state.foreignLock) driveState = 'locked';
  return {
    ...status,
    storage: volume.local ? 'local' : 'drive',
    writable: isVolumeWritable(volume),
    items: counts.items,
    chunks: counts.chunks,
    diskBytes: driveIndexBytes(volume.indexDir),
    lastIndexedAt: state.lastIndexedAt || null,
    writer: state.lastWriter || undefined,
    incompatible: state.incompatible || undefined,
    state: driveState,
    lock: state.foreignLock ? { owner: driveLockOwner(state.foreignLock), heartbeat: state.foreignLock.heartbeat || null } : undefined,
    error: state.error,
  };
}

/** Opts a drive in (indexed right away) or out (its index is deleted). */
export async function setSemanticSearchVolumeEnabled(root: string, enabled: boolean): Promise<{ error: string | null; status: SemanticSearchStatus }> {
  if (!enabled) {
    const state = drives.get(root);
    if (state) {
      // Stop writing before the folder goes away.
      state.closing = true;
      if (activeDriveRoot === root) activeDriveAbort = 'drive opted out';
    }
  }
  const error = await setVolumeEnabled(root, enabled);
  if (!error && enabled && drives.get(root) && isVolumeWritable(drives.get(root)!.volume)) {
    scheduleIndexing('drive-enabled', 0, { manual: true });
  }
  return { error, status: getSemanticSearchStatus() };
}

/** Deletes another machine's lock on a drive index, then updates it. */
export function breakSemanticSearchLock(root: string): { error: string | null; status: SemanticSearchStatus } {
  const state = drives.get(root);
  if (!state) return { error: `${root} is not an indexed drive`, status: getSemanticSearchStatus() };
  if (activeDriveRoot === root) return { error: 'This Mac is updating that index right now', status: getSemanticSearchStatus() };
  if (!breakDriveLock(state.volume.indexDir)) return { error: `Could not delete the lock in ${state.volume.indexDir}`, status: getSemanticSearchStatus() };
  console.log(`[SemanticSearch] Lock of ${root} broken by the user (was held by ${state.foreignLock ? driveLockOwner(state.foreignLock) : 'unknown'})`);
  state.foreignLock = null;
  state.error = undefined;
  if (isVolumeWritable(state.volume)) scheduleIndexing('lock-broken', 0, { manual: true });
  return { error: null, status: getSemanticSearchStatus() };
}

function syncDriveMonitor(config: SemanticSearchConfig): void {
  if (config.enabled) {
    startDriveMonitor({ dataDir: getRootDir(), writer: driveWriter(), onChange: syncDrives });
  } else {
    stopDriveMonitor();
  }
}

export function refreshSemanticSearchIndex(options: { force?: boolean } = {}): SemanticSearchStatus {
  if (options.force) {
    cancelIndexing();
    clearIndex();
  }
  scheduleIndexing(options.force ? 'manual-rebuild' : 'manual', 0, { manual: true });
  return getSemanticSearchStatus();
}

// ─── Search ─────────────────────────────────────────────────────────

export async function searchSemantic(
  query: string,
  options: {
    limit?: number;
    kinds?: SemanticItemKind[];
    /** Only results that stand out at least this much from the rest (z-score). */
    minZ?: number;
    /** From the main launcher search: skipped when that's turned off or the model isn't there, and never starts indexing. */
    launcher?: boolean;
  } = {}
): Promise<SemanticSearchResult[]> {
  const trimmed = String(query || '').trim();
  const config = getSemanticSearchConfig();
  if (!trimmed || !config.enabled) return [];
  if (options.launcher && (!config.showInLauncher || !isReadyToIndex(config))) return [];

  const items = loadIndex();
  if (!options.launcher && !indexingPromise && (!indexLastIndexedAt || Date.now() - indexLastIndexedAt > STALE_INDEX_MS)) {
    scheduleIndexing('search', 2000);
  }
  await ensureDrivesLoaded();
  const sources = [items];
  for (const state of drives.values()) {
    if (!state.closing && state.items && !state.incompatible) sources.push(state.items);
  }
  if (sources.every((source) => source.size === 0)) return [];

  const [[rawVector], centroid] = await Promise.all([
    embedTexts([`${QUERY_PREFIX}${trimmed}`], 'high'),
    getQueryCentroid(),
  ]);
  if (!rawVector) return [];
  const queryVector = centroid ? debiasQuery(rawVector, centroid, QUERY_DEBIAS_STRENGTH) : rawVector;

  const kinds = options.kinds && options.kinds.length > 0 ? new Set(options.kinds) : null;
  const terms = queryTerms(trimmed);
  const scored: Array<{ item: IndexedItem; score: number; row: number; z: number }> = [];
  for (const source of sources) {
    for (const item of source.values()) {
      if (item.scales.length === 0 || (kinds && !kinds.has(item.kind))) continue;
      const { score, row } = bestRowScore(queryVector, item.vectors, item.scales, EMBEDDING_DIM);
      scored.push({ item, score: score + titleBoost(item.title, terms), row, z: 0 });
    }
  }
  const zs = standoutScores(scored.map((entry) => entry.score));
  scored.forEach((entry, i) => { entry.z = zs[i]; });

  const minZ = Number.isFinite(options.minZ) ? Number(options.minZ) : null;
  const ranked = minZ === null ? scored : scored.filter((entry) => entry.z >= minZ);
  ranked.sort((a, b) => b.score - a.score);
  return ranked.slice(0, Math.max(1, Math.min(200, options.limit ?? 50))).map(({ item, score, row, z }) => ({
    id: item.key,
    kind: item.kind,
    title: item.title,
    subtitle: item.subtitle,
    preview: item.previews[row] || '',
    score: Math.round(score * 1000) / 1000,
    z: Math.round(z * 100) / 100,
    path: item.path,
    noteId: item.noteId,
    clipboardId: item.clipboardId,
    updatedAt: item.updatedAt,
  }));
}

// ─── Status & lifecycle ─────────────────────────────────────────────

export function getSemanticSearchStatus(): SemanticSearchStatus {
  const config = getSemanticSearchConfig();
  const counts = indexItems
    ? countIndex(indexItems)
    : indexCounts ?? (fs.existsSync(getIndexMetaPath()) ? countIndex(loadIndex()) : { items: 0, chunks: 0 });
  const textModel = getModelFileStatus(TEXT_MODELS[config.modelVariant]);
  return {
    config,
    volumes: config.enabled ? listVolumes().map(driveStatus) : [],
    helperAvailable: isHelperAvailable(),
    textModel,
    mediaModel: getModelFileStatus(MEDIA_MODEL),
    ready: config.enabled && textModel.state === 'downloaded' && isHelperAvailable(),
    index: {
      ...indexStatus,
      ...counts,
      lastIndexedAt: indexLastIndexedAt,
      pausedOnBattery: isPausedOnBattery(config) && indexStatus.state !== 'indexing',
    },
  };
}

/** Removes downloaded models and the index (frees all disk used by the feature). */
export function resetSemanticSearch(): SemanticSearchStatus {
  cancelIndexing();
  stopHelper();
  clearIndex();
  fs.rmSync(getModelsDir(), { recursive: true, force: true });
  downloadStatus.clear();
  updateSemanticSearchConfig({ enabled: false });
  return getSemanticSearchStatus();
}

function handlePowerConnected(): void {
  const config = getSemanticSearchConfig();
  if (!isReadyToIndex(config)) return;
  const stale = !indexLastIndexedAt || Date.now() - indexLastIndexedAt > BACKGROUND_REFRESH_MS;
  if (indexingDeferred || stale) scheduleIndexing('power-connected', POWER_CONNECTED_DELAY_MS);
}

function handlePowerDisconnected(): void {
  if (!getSemanticSearchConfig().onlyWhileCharging) return;
  if (scheduledIndexTimer) {
    clearTimeout(scheduledIndexTimer);
    scheduledIndexTimer = null;
    indexingDeferred = true;
  }
  // Pause a background run; embedded items are kept and the rest resumes on AC.
  if (indexingPromise && !activeIndexingManual && activeIndexingGeneration === indexingGeneration) {
    console.log('[SemanticSearch] On battery, pausing indexing until power is connected');
    cancelIndexing();
    indexingDeferred = true;
  }
}

export function initSemanticSearch(): void {
  const config = getSemanticSearchConfig();
  if (!fs.existsSync(getConfigPath())) {
    writeFileAtomic(getConfigPath(), JSON.stringify(config, null, 2));
  }
  powerMonitor.removeListener('on-ac', handlePowerConnected);
  powerMonitor.removeListener('on-battery', handlePowerDisconnected);
  powerMonitor.on('on-ac', handlePowerConnected);
  powerMonitor.on('on-battery', handlePowerDisconnected);
  if (refreshTimer) clearInterval(refreshTimer);
  refreshTimer = setInterval(() => {
    if (isReadyToIndex(getSemanticSearchConfig())) void runIndexing('background');
  }, BACKGROUND_REFRESH_MS);
  if (isReadyToIndex(config)) scheduleIndexing('startup', FIRST_REFRESH_DELAY_MS);
  syncDriveMonitor(config);
}

export function shutdownSemanticSearch(): void {
  cancelIndexing();
  if (refreshTimer) clearInterval(refreshTimer);
  if (scheduledIndexTimer) clearTimeout(scheduledIndexTimer);
  refreshTimer = null;
  scheduledIndexTimer = null;
  if (indexUnloadTimer) clearTimeout(indexUnloadTimer);
  indexUnloadTimer = null;
  powerMonitor.removeListener('on-ac', handlePowerConnected);
  powerMonitor.removeListener('on-battery', handlePowerDisconnected);
  stopDriveMonitor();
  try { saveIndex(); } catch {}
  stopHelper();
}
