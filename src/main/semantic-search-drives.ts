/**
 * semantic-search-drives.ts
 *
 * External and network drives that carry their own semantic index
 * (docs/semantic-index-format.md), shared with krunner-embed on Linux.
 *
 * - Drives under /Volumes and network shares are listed; nothing on a drive
 *   is read until the user opts it in. The opt-in is `volume.json` in the
 *   drive's `.semantic-index.noindex/` folder, so it follows the drive to
 *   other machines. Read-only drives keep it (and their index) locally.
 * - `lock.json` marks the machine updating a drive's index; a lock whose
 *   heartbeat is 15 minutes old belongs to a machine that died or lost the
 *   drive, and is taken over.
 *
 * File system and lock handling only; indexing and search live in
 * semantic-search.ts.
 */

import * as fs from 'fs';
import * as os from 'os';
import * as path from 'path';
import { randomUUID } from 'crypto';
import { execFile } from 'child_process';
import {
  DRIVE_INDEX_FOLDER,
  decodeDriveIndex,
  driveLockOwner,
  encodeDriveIndex,
  isDriveLockStale,
  isEligibleMacMount,
  isNetworkFs,
  parseDriveLock,
  parseMacMounts,
  type DecodedDriveIndex,
  type DriveIndexHeader,
  type DriveIndexItem,
  type DriveLockInfo,
} from './semantic-search-core';

const VOLUME_FILE = 'volume.json';
const LOCK_FILE = 'lock.json';
const INDEX_FILE = 'index.bin';
const REFRESH_DEBOUNCE_MS = 700;
const POLL_MS = 60 * 1000; // in case a /Volumes change notification is missed

export interface DriveVolume {
  /** Mount point. */
  root: string;
  label: string;
  fsType: string;
  source: string;
  /** Names the local folder of a read-only drive. */
  key: string;
  /** st_dev while mounted; changes with every mount. */
  dev: number;
  network: boolean;
  readOnly: boolean;
  bytesTotal: number;
  enabled: boolean;
  /** The index is kept on this Mac (the drive is read-only). */
  local: boolean;
  /** Where index.bin and lock.json are; empty unless enabled. */
  indexDir: string;
}

export function isVolumeWritable(volume: DriveVolume): boolean {
  return volume.enabled && (volume.local || !volume.readOnly);
}

let dataDir = '';
let writer = '';
let volumes: DriveVolume[] = [];
let onChange: (() => void) | null = null;
let watcher: fs.FSWatcher | null = null;
let pollTimer: NodeJS.Timeout | null = null;
let debounceTimer: NodeJS.Timeout | null = null;
let refreshing: Promise<void> | null = null;
let refreshAgain = false;

function localDir(volume: DriveVolume): string {
  return path.join(dataDir, 'volumes', volume.key);
}

function sanitizeKey(text: string): string {
  return text.replace(/[^A-Za-z0-9.-]/g, '_').slice(0, 120);
}

function readOptIn(volume: DriveVolume): void {
  volume.enabled = false;
  volume.local = false;
  volume.indexDir = '';
  const onDrive = path.join(volume.root, DRIVE_INDEX_FOLDER);
  const local = localDir(volume);
  for (const dir of [onDrive, local]) {
    try {
      const json = JSON.parse(fs.readFileSync(path.join(dir, VOLUME_FILE), 'utf8'));
      if (json?.format !== 1 || json?.enabled !== true) continue;
      volume.enabled = true;
      volume.local = dir === local;
      volume.indexDir = dir;
      return;
    } catch {}
  }
}

function runMount(): Promise<string> {
  return new Promise((resolve) => {
    execFile('/sbin/mount', [], { timeout: 10_000 }, (error, stdout) => resolve(error ? '' : String(stdout)));
  });
}

async function readVolumes(): Promise<DriveVolume[]> {
  const out: DriveVolume[] = [];
  const seen = new Set<number>();
  for (const mount of parseMacMounts(await runMount())) {
    if (!isEligibleMacMount(mount)) continue;
    let dev: number;
    try {
      dev = (await fs.promises.stat(mount.mountPoint)).dev;
    } catch {
      continue;
    }
    if (seen.has(dev)) continue;
    seen.add(dev);
    // Known drives are not touched again (no wake-ups of sleeping disks or
    // hung shares); only a new mount is looked at.
    const known = volumes.find((v) => v.root === mount.mountPoint && v.dev === dev);
    if (known) {
      out.push({ ...known, readOnly: mount.readOnly });
      continue;
    }
    let bytesTotal = 0;
    try {
      const fsStats = await fs.promises.statfs(mount.mountPoint);
      bytesTotal = fsStats.blocks * fsStats.bsize;
    } catch {}
    const network = isNetworkFs(mount.fsType);
    const label = path.basename(mount.mountPoint);
    const volume: DriveVolume = {
      root: mount.mountPoint,
      label,
      fsType: mount.fsType,
      source: mount.source,
      key: sanitizeKey(network ? `net-${mount.source}` : `${label}-${bytesTotal}`),
      dev,
      network,
      readOnly: mount.readOnly,
      bytesTotal,
      enabled: false,
      local: false,
      indexDir: '',
    };
    readOptIn(volume);
    out.push(volume);
  }
  return out.sort((a, b) => a.root.localeCompare(b.root));
}

function sameVolumes(a: DriveVolume[], b: DriveVolume[]): boolean {
  return a.length === b.length && a.every((v, i) =>
    v.root === b[i].root && v.dev === b[i].dev && v.enabled === b[i].enabled && v.indexDir === b[i].indexDir && v.readOnly === b[i].readOnly);
}

export function refreshVolumes(): Promise<void> {
  if (refreshing) {
    refreshAgain = true;
    return refreshing;
  }
  refreshing = (async () => {
    do {
      refreshAgain = false;
      const next = await readVolumes();
      const changed = !sameVolumes(next, volumes);
      volumes = next;
      if (changed) onChange?.();
    } while (refreshAgain);
  })().finally(() => {
    refreshing = null;
  });
  return refreshing;
}

function scheduleRefresh(): void {
  if (debounceTimer) clearTimeout(debounceTimer);
  debounceTimer = setTimeout(() => {
    debounceTimer = null;
    void refreshVolumes();
  }, REFRESH_DEBOUNCE_MS);
}

export function startDriveMonitor(options: { dataDir: string; writer: string; onChange: () => void }): void {
  dataDir = options.dataDir;
  writer = options.writer;
  onChange = options.onChange;
  stopDriveMonitor();
  try {
    watcher = fs.watch('/Volumes', () => scheduleRefresh());
    watcher.on('error', () => {});
  } catch {}
  pollTimer = setInterval(() => { void refreshVolumes(); }, POLL_MS);
  void refreshVolumes();
}

export function stopDriveMonitor(): void {
  watcher?.close();
  watcher = null;
  if (pollTimer) clearInterval(pollTimer);
  pollTimer = null;
  if (debounceTimer) clearTimeout(debounceTimer);
  debounceTimer = null;
}

export function listVolumes(): DriveVolume[] {
  return volumes.map((v) => ({ ...v }));
}

export function findVolume(root: string): DriveVolume | undefined {
  return volumes.find((v) => v.root === root);
}

/** Still the same filesystem at `root`? (Unmounting removes the /Volumes folder.) */
export function isVolumeMounted(root: string, dev: number): boolean {
  try {
    return fs.statSync(root).dev === dev;
  } catch {
    return false;
  }
}

function writeFileAtomic(filePath: string, data: string | Uint8Array): void {
  const temporary = `${filePath}.${randomUUID()}.tmp`;
  try {
    fs.writeFileSync(temporary, data);
    fs.renameSync(temporary, filePath);
  } catch (error) {
    fs.rmSync(temporary, { force: true });
    throw error;
  }
}

// Removes an index folder: only the files the format defines (and macOS's
// AppleDouble/.DS_Store litter), then the folder itself if it ended up empty.
function removeIndexFolder(dir: string): boolean {
  let names: string[];
  try {
    names = fs.readdirSync(dir);
  } catch {
    return true;
  }
  for (const name of names) {
    if (name === INDEX_FILE || name === VOLUME_FILE || name === LOCK_FILE || name.endsWith('.tmp') || name.startsWith(`${INDEX_FILE}.`)
      || name.startsWith('._') || name === '.DS_Store') {
      try { fs.rmSync(path.join(dir, name), { force: true }); } catch {}
    }
  }
  try { fs.rmdirSync(dir); } catch {}
  return !fs.existsSync(path.join(dir, INDEX_FILE));
}

/**
 * Turns indexing of a drive on (creates volume.json) or off (deletes its
 * index folder; the files on the drive are untouched). Returns an error.
 */
export async function setVolumeEnabled(root: string, enabled: boolean): Promise<string | null> {
  await refreshVolumes();
  const volume = volumes.find((v) => v.root === root);
  if (!volume) return `${root} is not an external or network drive`;
  const onDrive = path.join(volume.root, DRIVE_INDEX_FOLDER);
  const local = localDir(volume);
  const reread = () => {
    readOptIn(volume);
    onChange?.();
  };
  if (!enabled) {
    const removed = removeIndexFolder(onDrive) && removeIndexFolder(local);
    try { fs.rmSync(path.join(volume.root, `._${DRIVE_INDEX_FOLDER}`), { force: true }); } catch {} // AppleDouble twin on FAT/exFAT
    reread();
    return removed ? null : `Could not remove ${onDrive}`;
  }
  readOptIn(volume); // maybe opted in elsewhere meanwhile
  if (volume.enabled) {
    onChange?.();
    return null;
  }
  const json = JSON.stringify({
    format: 1,
    id: randomUUID(),
    enabled: true,
    createdAt: new Date().toISOString().replace(/\.\d{3}Z$/, 'Z'),
    createdBy: writer,
  }, null, 2);
  // On the drive when we can write there; else (read-only) on this Mac.
  if (!volume.readOnly) {
    try {
      fs.mkdirSync(onDrive, { recursive: true });
      writeFileAtomic(path.join(onDrive, VOLUME_FILE), json);
      reread();
      return null;
    } catch {
      try { fs.rmdirSync(onDrive); } catch {}
    }
  }
  try {
    fs.mkdirSync(local, { recursive: true });
    writeFileAtomic(path.join(local, VOLUME_FILE), json);
  } catch (error) {
    return `Could not write ${path.join(local, VOLUME_FILE)}: ${error instanceof Error ? error.message : String(error)}`;
  }
  reread();
  return null;
}

// ─── Lock ───────────────────────────────────────────────────────────

function isProcessAlive(pid: number): boolean {
  if (pid <= 0) return false;
  try {
    process.kill(pid, 0);
    return true;
  } catch (error: any) {
    return error?.code === 'EPERM';
  }
}

export function readDriveLock(dir: string): DriveLockInfo | null {
  const file = path.join(dir, LOCK_FILE);
  try {
    const text = fs.readFileSync(file, 'utf8').slice(0, 64 * 1024);
    return parseDriveLock(text, fs.statSync(file).mtimeMs);
  } catch {
    return null;
  }
}

/** Deletes the lock file whoever holds it (the user's "Break lock"). */
export function breakDriveLock(dir: string): boolean {
  const file = path.join(dir, LOCK_FILE);
  try { fs.rmSync(file, { force: true }); } catch {}
  return !fs.existsSync(file);
}

export class DriveLock {
  private token = '';
  private since = '';
  private readonly host = os.hostname();
  error = '';

  constructor(private readonly dir: string, private readonly app: string) {}

  get held(): boolean {
    return Boolean(this.token);
  }

  private encode(): string {
    return JSON.stringify({
      token: this.token,
      host: this.host,
      pid: process.pid,
      app: this.app,
      since: this.since,
      heartbeat: new Date().toISOString().replace(/\.\d{3}Z$/, 'Z'),
    });
  }

  /** Takes the lock, or a stale one over. `holder` is set when another machine has it. */
  acquire(): { result: 'acquired' | 'held' | 'error'; holder?: DriveLockInfo } {
    if (this.held) return { result: 'acquired' };
    this.error = '';
    const file = path.join(this.dir, LOCK_FILE);
    let holder: DriveLockInfo | undefined;
    for (let attempt = 0; attempt < 2; attempt++) {
      this.token = randomUUID();
      this.since = new Date().toISOString().replace(/\.\d{3}Z$/, 'Z');
      let fd = -1;
      try {
        fd = fs.openSync(file, 'wx'); // O_CREAT | O_EXCL
      } catch (error: any) {
        if (error?.code !== 'EEXIST') {
          this.token = '';
          this.error = error instanceof Error ? error.message : String(error);
          return { result: 'error' };
        }
      }
      if (fd >= 0) {
        let written = false;
        try {
          fs.writeSync(fd, this.encode());
          try { fs.fsyncSync(fd); } catch {}
          written = true;
        } catch (error) {
          this.error = error instanceof Error ? error.message : String(error);
        } finally {
          fs.closeSync(fd);
        }
        if (!written) {
          try { fs.rmSync(file, { force: true }); } catch {}
          this.token = '';
          return { result: 'error' };
        }
        // Read it back: some network filesystems don't honour O_EXCL.
        if (this.verify()) return { result: 'acquired' };
      }
      this.token = '';
      holder = readDriveLock(this.dir) ?? undefined;
      if (holder && !isDriveLockStale(holder, Date.now(), this.host, process.pid, isProcessAlive)) return { result: 'held', holder };
      console.log(`[SemanticSearch] Taking over a stale lock of ${holder ? driveLockOwner(holder) : 'an unknown app'} in ${this.dir}`);
      try { fs.rmSync(file, { force: true }); } catch {}
    }
    return { result: 'held', holder };
  }

  /** True while lock.json still carries our token (checked before every save). */
  verify(): boolean {
    return this.held && readDriveLock(this.dir)?.token === this.token;
  }

  /** Refreshes the heartbeat; false if another machine took the lock over. */
  heartbeat(): boolean {
    if (!this.held) return false;
    if (!this.verify()) {
      console.warn(`[SemanticSearch] Lock in ${this.dir} was taken over by another machine`);
      this.token = '';
      return false;
    }
    try {
      writeFileAtomic(path.join(this.dir, LOCK_FILE), this.encode());
    } catch {}
    return true;
  }

  release(): void {
    if (this.verify()) {
      try { fs.rmSync(path.join(this.dir, LOCK_FILE), { force: true }); } catch {}
    }
    this.token = '';
  }
}

// ─── Index files ────────────────────────────────────────────────────

export function driveIndexStamp(dir: string): string {
  try {
    const stat = fs.statSync(path.join(dir, INDEX_FILE));
    return `${stat.mtimeMs}:${stat.size}`;
  } catch {
    return '';
  }
}

export function driveIndexBytes(dir: string): number {
  try {
    return fs.statSync(path.join(dir, INDEX_FILE)).size;
  } catch {
    return 0;
  }
}

/** Reads a drive index; a missing file is an empty, compatible index. */
export async function readDriveIndex(dir: string): Promise<DecodedDriveIndex & { stamp: string }> {
  const stamp = driveIndexStamp(dir);
  let data: Buffer;
  try {
    data = await fs.promises.readFile(path.join(dir, INDEX_FILE));
  } catch (error: any) {
    if (error?.code === 'ENOENT') return { ok: true, header: { quant: '', lastIndexedAt: 0, writer: '' }, items: [], stamp: '' };
    return { ok: false, incompatible: false, error: error instanceof Error ? error.message : String(error), stamp };
  }
  return { ...decodeDriveIndex(new Uint8Array(data.buffer, data.byteOffset, data.byteLength)), stamp };
}

export function writeDriveIndex(dir: string, header: DriveIndexHeader, items: DriveIndexItem[]): string {
  fs.mkdirSync(dir, { recursive: true });
  writeFileAtomic(path.join(dir, INDEX_FILE), encodeDriveIndex(header, items));
  return driveIndexStamp(dir);
}
