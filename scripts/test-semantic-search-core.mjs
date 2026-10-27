#!/usr/bin/env node

import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { importTs } from './lib/ts-import.mjs';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const {
  CHUNK_CHARS,
  CHUNK_OVERLAP,
  MAX_CHUNKS_PER_ITEM,
  chunkText,
  quantizeRows,
  bestRowScore,
  titleBoost,
  queryTerms,
  debiasQuery,
  meanDirection,
  standoutScores,
  QUERY_CENTROID_PROBES,
  encodeDriveIndex,
  decodeDriveIndex,
  sameMtime,
  parseDriveLock,
  isDriveLockStale,
  driveLockOwner,
  parseMacMounts,
  isEligibleMacMount,
} = await importTs(path.join(root, 'src/main/semantic-search-core.ts'));

const DIM = 256;

// Deterministic pseudo-random unit vectors (mulberry32).
function rng(seed) {
  return () => {
    seed |= 0;
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function unitVector(random) {
  const v = Float32Array.from({ length: DIM }, () => random() * 2 - 1);
  const norm = Math.hypot(...v);
  return v.map((x) => x / norm);
}

function dot(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}

test('chunkText returns nothing for blank input and one chunk for short text', () => {
  assert.deepEqual(chunkText('   \n\n  '), []);
  assert.deepEqual(chunkText('  Hello   world\r\n'), ['Hello world']);
});

test('chunkText splits long text into overlapping, bounded chunks on word boundaries', () => {
  const words = Array.from({ length: 2000 }, (_, i) => `word${i}`);
  const chunks = chunkText(words.join(' '));

  assert.equal(chunks.length, MAX_CHUNKS_PER_ITEM);
  for (const chunk of chunks) {
    assert.ok(chunk.length <= CHUNK_CHARS, `chunk too long: ${chunk.length}`);
    assert.ok(chunk.length >= CHUNK_CHARS * 0.5, `chunk too short: ${chunk.length}`);
    // No chunk starts or ends mid-word.
    assert.match(chunk, /^\S*word\d+/);
    assert.match(chunk, /word\d+$/);
  }
  // Consecutive chunks overlap by roughly CHUNK_OVERLAP characters.
  const tail = chunks[0].slice(-CHUNK_OVERLAP + 20);
  assert.ok(chunks[1].includes(tail.trim().split(' ').pop()), 'expected overlap between chunks');
});

test('chunkText prefers paragraph breaks', () => {
  const para = (n) => `${'lorem ipsum '.repeat(n).trim()}.`;
  const text = `${para(70)}\n\n${para(70)}`;
  const [first] = chunkText(text);
  assert.ok(first.endsWith('.'), 'first chunk should end at the paragraph break');
});

test('int8 quantization preserves cosine scores', () => {
  const random = rng(42);
  const docs = Array.from({ length: 200 }, () => unitVector(random));
  const query = unitVector(random);
  const { vectors, scales } = quantizeRows(docs, DIM);

  let maxError = 0;
  docs.forEach((doc, row) => {
    const exact = dot(query, doc);
    const quantized = bestRowScore(
      query,
      vectors.subarray(row * DIM, (row + 1) * DIM),
      scales.subarray(row, row + 1),
      DIM
    ).score;
    maxError = Math.max(maxError, Math.abs(exact - quantized));
  });
  assert.ok(maxError < 0.01, `quantization error too large: ${maxError}`);
});

test('bestRowScore picks the best-matching passage of an item', () => {
  const random = rng(7);
  const query = unitVector(random);
  const rows = [unitVector(random), query, unitVector(random)];
  const { vectors, scales } = quantizeRows(rows, DIM);
  const { score, row } = bestRowScore(query, vectors, scales, DIM);
  assert.equal(row, 1);
  assert.ok(score > 0.99, `expected near-1 self similarity, got ${score}`);
});

test('quantizeRows handles zero vectors without NaN', () => {
  const { vectors, scales } = quantizeRows([new Float32Array(DIM)], DIM);
  assert.ok(vectors.every((v) => v === 0));
  assert.ok(Number.isFinite(scales[0]));
});

test('titleBoost rewards title matches and ignores short terms', () => {
  assert.deepEqual(queryTerms('a red Car on the road'), ['red', 'car', 'the', 'road']);
  assert.equal(titleBoost('anything', []), 0);
  assert.equal(titleBoost('Red car.jpg', queryTerms('red car')), 0.04);
  assert.equal(titleBoost('Red bike.jpg', queryTerms('red car')), 0.02);
});

// ─── Drive indexes (shared with krunner-embed on Linux) ─────────────

// The same three items are written by krunner-embed's tests/test_core.cpp;
// each side checks that it reads the other's file.
function fixtureItems() {
  const vectorRow = (r) => Int8Array.from({ length: DIM }, (_, i) => ((i * (r + 3) + r * 7) % 255) - 127);
  const rows = (n) => {
    const out = new Int8Array(n * DIM);
    for (let r = 0; r < n; r++) out.set(vectorRow(r), r * DIM);
    return out;
  };
  return [
    {
      path: 'Docs/cafe\u0301.md', kind: 'document', mtime: 1700000000123, size: 4242, hash: new Uint8Array(20).fill(0x68),
      scales: Float32Array.of(0.01, 0.02), vectors: rows(2), previews: ['passage 0 — ünïcödé', 'passage 1 — ünïcödé'],
    },
    {
      path: 'photo.jpg', kind: 'image', mtime: 1700000000456, size: 99999, hash: new Uint8Array(20).fill(0xab),
      scales: Float32Array.of(0.005), vectors: rows(1), previews: [''],
    },
    { path: 'broken.pdf', kind: 'document', mtime: 1700000000789, size: 1, hash: new Uint8Array(0), scales: new Float32Array(0), vectors: new Int8Array(0), previews: [] },
  ];
}

function assertFixture(decoded, writer) {
  assert.equal(decoded.ok, true, decoded.error);
  assert.equal(decoded.header.writer, writer);
  assert.equal(decoded.header.lastIndexedAt, 1793001234567);
  const byPath = new Map(decoded.items.map((item) => [item.path, item]));
  for (const expected of fixtureItems()) {
    const item = byPath.get(expected.path);
    assert.ok(item, `missing ${expected.path}`);
    assert.equal(item.kind, expected.kind);
    assert.equal(item.mtime, expected.mtime);
    assert.equal(item.size, expected.size);
    assert.deepEqual([...item.hash], [...expected.hash]);
    assert.deepEqual([...item.scales], [...expected.scales]);
    assert.deepEqual([...item.vectors], [...expected.vectors]);
    assert.deepEqual(item.previews, expected.previews);
  }
  assert.equal(decoded.items.length, 3);
}

test('debiasQuery lets a specific match beat generic "hub" items', () => {
  const basis = (i) => Float32Array.from({ length: DIM }, (_, k) => (k === i ? 1 : 0));
  const mix = (...parts) => {
    const v = new Float32Array(DIM);
    for (const [w, b] of parts) for (let k = 0; k < DIM; k++) v[k] += w * b[k];
    const norm = Math.hypot(...v);
    return v.map((x) => x / norm);
  };
  // Every query shares a generic direction (0); short snippets sit close to it.
  const generic = basis(0);
  const probes = Array.from({ length: 12 }, (_, i) => mix([0.75, generic], [0.66, basis(10 + i)]));
  const centroid = meanDirection(probes, DIM);
  assert.ok(dot(centroid, generic) > 0.95);
  assert.ok(Math.abs(Math.hypot(...centroid) - 1) < 1e-5);

  const query = mix([0.75, generic], [0.66, basis(1)]);
  const hub = mix([0.95, generic], [0.3, basis(2)]);
  const target = mix([0.1, generic], [0.6, basis(1)], [0.79, basis(3)]);
  assert.ok(dot(query, hub) > dot(query, target), 'the hub wins uncorrected');

  const corrected = debiasQuery(query, centroid, 0.8);
  assert.ok(Math.abs(Math.hypot(...corrected) - 1) < 1e-5);
  assert.ok(dot(corrected, target) > dot(corrected, hub) + 0.3, 'the real match wins corrected');
  assert.deepEqual(Array.from(debiasQuery(query, centroid, 0)).map((x) => x.toFixed(5)), Array.from(query).map((x) => x.toFixed(5)));
  assert.ok(QUERY_CENTROID_PROBES.length >= 20 && new Set(QUERY_CENTROID_PROBES).size === QUERY_CENTROID_PROBES.length);
});

test('standoutScores: a clear winner stands out, a flat field does not', () => {
  const random = rng(7);
  const field = Array.from({ length: 200 }, () => 0.3 + random() * 0.05);
  const z = standoutScores([...field, 0.62]);
  assert.ok(z[200] > 10);
  assert.ok(Math.max(...z.slice(0, 200)) < 3);
  assert.ok(standoutScores([0.4, 0.4, 0.4, 0.4]).every((value) => Math.abs(value) < 1e-9));
  assert.deepEqual(standoutScores([0.9, 0.1]), [10, 10]);
  assert.deepEqual(standoutScores([]), []);
});

test('drive index: encode/decode round trip', () => {
  const header = { quant: 'UD-Q4_K_XL', lastIndexedAt: 1793001234567, writer: 'SuperCmd fixture' };
  const bytes = encodeDriveIndex(header, fixtureItems());
  assert.equal(new TextDecoder().decode(bytes.subarray(0, 8)), 'SEMINDEX');
  assert.deepEqual([...bytes.subarray(8, 12)], [1, 0, 0, 0], 'version is u32 little-endian 1');
  assertFixture(decodeDriveIndex(bytes), 'SuperCmd fixture');
  if (process.env.SC_WRITE_FIXTURE) fs.writeFileSync(process.env.SC_WRITE_FIXTURE, bytes);
});

test('drive index: reads the file written by krunner-embed (Linux)', () => {
  const file = path.join(root, 'scripts/fixtures/semantic-index/krunner-embed.bin');
  assertFixture(decodeDriveIndex(new Uint8Array(fs.readFileSync(file))), 'krunner-embed fixture');
});

test('drive index: damaged, foreign and unsafe files', () => {
  const header = { quant: 'Q8_0', lastIndexedAt: 1, writer: 'x' };
  const bytes = encodeDriveIndex(header, fixtureItems());
  const truncated = decodeDriveIndex(bytes.subarray(0, bytes.length - 50));
  assert.equal(truncated.ok, false);
  assert.equal(truncated.incompatible, false);
  const other = decodeDriveIndex(new TextEncoder().encode(new TextDecoder().decode(bytes).replace('embeddinggemma-2', 'embeddinggemma-9')));
  assert.equal(other.ok, false);
  assert.equal(other.incompatible, true);
  assert.equal(decodeDriveIndex(new TextEncoder().encode('not an index at all')).ok, false);
  // Paths that would leave the drive are neither written nor read.
  const [item] = fixtureItems();
  const unsafe = decodeDriveIndex(encodeDriveIndex(header, [
    { ...item, path: '../outside.md' }, { ...item, path: '/etc/passwd' }, { ...item, path: 'ok/inside.md' },
  ]));
  assert.deepEqual(unsafe.items.map((i) => i.path), ['ok/inside.md']);
});

test('sameMtime tolerates FAT rounding and time-zone shifts on drives', () => {
  const h = 3600 * 1000;
  const t = 1700000000000;
  assert.ok(sameMtime(t, t, false));
  assert.ok(!sameMtime(t, t + 1, false));
  assert.ok(sameMtime(t, t + 1999, true));
  assert.ok(!sameMtime(t, t + 2500, true));
  assert.ok(sameMtime(t, t + h, true));
  assert.ok(sameMtime(t, t - 3 * h + 1500, true));
  assert.ok(sameMtime(t, t + 14 * h, true));
  assert.ok(!sameMtime(t, t + 15 * h, true));
  assert.ok(!sameMtime(t, t + h / 2, true));
});

test('drive locks: live, stale, own machine, unparseable', () => {
  const now = Date.parse('2026-10-27T12:00:00Z');
  const lock = (host, pid, minutesAgo) => {
    const when = new Date(now - minutesAgo * 60_000).toISOString();
    return parseDriveLock(JSON.stringify({ token: 't', host, pid, app: 'krunner-embed 0.2.0', since: when, heartbeat: when }), now);
  };
  const alive = (pid) => pid === 100;
  assert.equal(driveLockOwner(lock('ege-Desktop', 1, 1)), 'krunner-embed 0.2.0 on ege-Desktop');
  assert.ok(!isDriveLockStale(lock('ege-Desktop', 1, 14), now, 'mac', 42, alive));
  assert.ok(isDriveLockStale(lock('ege-Desktop', 1, 16), now, 'mac', 42, alive));
  assert.ok(isDriveLockStale(lock('ege-Desktop', 1, 5 * 60), now, 'mac', 42, alive), 'the "5 hours ago" case');
  assert.ok(isDriveLockStale(lock('ege-Desktop', 1, -3 * 24 * 60), now, 'mac', 42, alive), 'broken clock');
  assert.ok(isDriveLockStale(lock('mac', 7, 1), now, 'mac', 42, alive), 'this machine, process gone');
  assert.ok(!isDriveLockStale(lock('mac', 100, 1), now, 'mac', 42, alive), 'this machine, process alive');
  assert.ok(isDriveLockStale(lock('mac', 42, 1), now, 'mac', 42, alive), 'left over by our own pid');
  assert.ok(!isDriveLockStale(parseDriveLock('{"tok', now - 60_000), now, 'mac', 42, alive), 'half-written, fresh');
  assert.ok(isDriveLockStale(parseDriveLock('{"tok', now - 3_600_000), now, 'mac', 42, alive), 'half-written, old');
});

test('mounted volumes: external, network and disk images count; system mounts do not', () => {
  const mounts = parseMacMounts([
    '/dev/disk3s1s1 on / (apfs, sealed, local, read-only, journaled)',
    'devfs on /dev (devfs, local, nobrowse)',
    '/dev/disk3s5 on /System/Volumes/Data (apfs, local, journaled, nobrowse, protect)',
    'map auto_home on /System/Volumes/Data/home (autofs, automounted, nobrowse)',
    '/dev/disk4s1 on /Volumes/My Stick (exfat, local, nodev, nosuid, noowners, noatime, fskit)',
    '/dev/disk5s1 on /Volumes/CD (cd9660, local, nodev, nosuid, read-only, noowners)',
    '//ege@nas.local/share on /Volumes/share (smbfs, nodev, nosuid, mounted by ege)',
    '/dev/disk6s2 on /Volumes/.timemachine/abc (apfs, local, nobrowse)',
    '/dev/disk7s1 on /Volumes/com.apple.TimeMachine.localsnapshots/x (apfs, local, read-only)',
    '/dev/disk3s3 on /Volumes/Recovery (apfs, local, journaled, nobrowse)',
  ].join('\n'));
  assert.equal(mounts.length, 10);
  assert.equal(mounts[4].mountPoint, '/Volumes/My Stick');
  assert.equal(mounts[4].fsType, 'exfat');
  assert.equal(mounts[5].readOnly, true);
  assert.deepEqual(mounts.filter(isEligibleMacMount).map((m) => m.mountPoint), ['/Volumes/My Stick', '/Volumes/CD', '/Volumes/share']);
});
