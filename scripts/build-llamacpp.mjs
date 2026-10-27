#!/usr/bin/env node

import { existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync } from 'fs';
import { createHash } from 'crypto';
import path from 'path';
import { tmpdir } from 'os';
import { spawnSync } from 'child_process';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const repoRoot = path.resolve(__dirname, '..');

// b11899 is the first release line with EmbeddingGemma 2 support
// (ggml-org/llama.cpp#30054) plus the follow-up pooled-embedding speedups.
const llamaVersion = 'b11899';
const frameworkUrl = `https://github.com/ggml-org/llama.cpp/releases/download/${llamaVersion}/llama-${llamaVersion}-xcframework.zip`;
// SHA-256 of the xcframework zip, computed from the b11899 release asset.
// Update this when bumping llamaVersion.
const frameworkSha256 = '500a6b8953d10542514daccccd8261e4522e3275692984aa53a380475ee0e625';

const distNativeDir = path.join(repoRoot, 'dist', 'native');
const runtimeDir = path.join(distNativeDir, 'llama-runtime');
const frameworkDir = path.join(runtimeDir, 'llama.framework');
const versionStampPath = path.join(runtimeDir, 'VERSION');
const serverSource = path.join(repoRoot, 'src', 'native', 'embedding-server.swift');
const serverBinary = path.join(distNativeDir, 'embedding-server');

function run(command, args, options = {}) {
  const result = spawnSync(command, args, {
    stdio: 'inherit',
    ...options,
  });
  if (result.status !== 0) {
    throw new Error(`${command} exited with status ${result.status ?? 'unknown'}`);
  }
}

function prepareFramework() {
  const tempRoot = mkdtempSync(path.join(tmpdir(), 'supercmd-llamacpp-'));
  const archivePath = path.join(tempRoot, 'llama-xcframework.zip');
  const extractDir = path.join(tempRoot, 'extract');

  mkdirSync(distNativeDir, { recursive: true });
  mkdirSync(extractDir, { recursive: true });

  console.log(`[llama.cpp] Downloading macOS framework (${llamaVersion})`);
  run('curl', ['-L', frameworkUrl, '-o', archivePath]);

  const fileBuffer = readFileSync(archivePath);
  const actualHash = createHash('sha256').update(fileBuffer).digest('hex');
  if (actualHash !== frameworkSha256) {
    throw new Error(
      `[llama.cpp] Integrity check failed!\n` +
      `  Expected SHA-256: ${frameworkSha256}\n` +
      `  Actual SHA-256:   ${actualHash}\n` +
      `  The downloaded file does not match the pinned hash. ` +
      `This could indicate a corrupted download or a compromised release asset.`
    );
  }
  console.log('[llama.cpp] Integrity check passed (SHA-256)');

  console.log('[llama.cpp] Extracting framework');
  run('unzip', ['-q', '-o', archivePath, '-d', extractDir]);

  const sourceFrameworkDir = path.join(
    extractDir,
    'build-apple',
    'llama.xcframework',
    'macos-arm64_x86_64',
    'llama.framework'
  );
  if (!existsSync(sourceFrameworkDir)) {
    throw new Error('Downloaded llama.cpp archive did not contain the macOS framework slice');
  }

  rmSync(runtimeDir, { recursive: true, force: true });
  mkdirSync(runtimeDir, { recursive: true });
  // Use cp -a instead of cpSync to preserve relative symlinks in the framework bundle.
  // Node's cpSync converts relative symlinks to absolute paths, which breaks codesign.
  run('cp', ['-a', sourceFrameworkDir, frameworkDir]);
  run('sh', ['-c', `printf %s ${llamaVersion} > "${versionStampPath}"`]);
  rmSync(tempRoot, { recursive: true, force: true });
}

function buildEmbeddingServer() {
  const moduleCacheDir = path.join(tmpdir(), 'supercmd-swift-module-cache');
  mkdirSync(moduleCacheDir, { recursive: true });

  console.log('[llama.cpp] Building embedding-server');
  run('swiftc', [
    '-O',
    '-module-cache-path', moduleCacheDir,
    '-F', runtimeDir,
    '-framework', 'llama',
    '-framework', 'AVFoundation',
    '-framework', 'AppKit',
    '-framework', 'PDFKit',
    '-Xlinker', '-rpath',
    '-Xlinker', '@executable_path/llama-runtime',
    '-o', serverBinary,
    serverSource,
  ]);
}

try {
  const moduleMapPath = path.join(frameworkDir, 'Modules', 'module.modulemap');
  const stampedVersion = existsSync(versionStampPath) ? readFileSync(versionStampPath, 'utf8').trim() : '';
  if (!existsSync(moduleMapPath) || stampedVersion !== llamaVersion) {
    prepareFramework();
  } else {
    console.log('[llama.cpp] Using existing macOS framework');
  }
  buildEmbeddingServer();
  console.log('[llama.cpp] Ready');
} catch (error) {
  console.error('[llama.cpp] Build failed:', error instanceof Error ? error.message : error);
  process.exit(1);
}
