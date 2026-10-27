# Semantic index format, version 1

Shared by krunner-embed (Linux) and SuperCmd (macOS); a copy of krunner-embed's
docs/index-format.md. Readers and writers: `src/main/semantic-search-core.ts`
(`encodeDriveIndex` / `decodeDriveIndex`) and `src/main/semantic-search-drives.ts`. An index written by
either can be read and updated by the other, including an index stored on a
removable or network drive.

## Compatibility rule

Embeddings are only comparable if they come from the same model, prompts
and chunking. Version 1 means:

| | |
|---|---|
| Model | EmbeddingGemma 2 (any GGUF quantization; the writer records which) |
| Dimensions | 256 (Matryoshka truncation of the 768-dim output, re-normalized) |
| Document prompt | `title: <file name> \| text: <passage>` |
| Query prompt | `task: search result \| query: <query>` |
| Media | embedded without a prompt; video = up to 8 frames, audio = first 60 s |
| Chunking | 1200 characters, 150 overlap, at most 8 passages per file (see `chunkText`) |
| Vectors | int8 per row with one float32 scale per row (`value = int8 * scale`) |

A reader that finds a different `model`, `dim`, `prompts` or `chunk` value
in the header must not mix that index with its own. It can ignore it, or
rebuild it if it is allowed to write there.

## Where indexes live

- **Home index**: in the app's data directory. Paths are relative to `/`.
- **Drive index**: `<mount point>/.semantic-index.noindex/` on the drive.
  The `.noindex` suffix keeps Spotlight out, and the leading dot keeps Baloo
  and file managers out. Paths are relative to the mount point, which differs
  between machines and OSes (`/Volumes/X` vs `/run/media/user/X`).
- **Read-only drive**: same files in the app's data directory, under
  `volumes/<volume key>/`, where the key is the filesystem UUID (or the
  network share's address).

A drive folder contains:

| File | Contents |
|---|---|
| `volume.json` | `{"format": 1, "id": "<random UUID>", "enabled": true, "createdAt": "<ISO 8601>", "createdBy": "<app> on <host>"}`. Its presence with `enabled: true` is the opt-in; deleting the folder opts out. |
| `index.bin` | the index (below), replaced atomically: write a temporary file in the same folder (e.g. `index.bin.<random>`), then rename it over `index.bin`. Readers ignore other files. |
| `lock.json` | present while some machine is updating the index (below) |

Other files in the folder are ignored (macOS adds `._*` AppleDouble files and
`.DS_Store` on FAT/exFAT drives). Opting out deletes the folder.

## Lock

Only the holder of `lock.json` writes `index.bin`. Readers never need it.

```json
{"token": "<random UUID>", "host": "ege-Desktop", "pid": 12345, "app": "krunner-embed 0.2.0",
 "since": "2026-10-27T10:00:00Z", "heartbeat": "2026-10-27T10:05:00Z"}
```

- **Acquire**: create the file exclusively (`O_CREAT | O_EXCL`), then read it
  back and check the token (some network filesystems don't honour `O_EXCL`).
- **Heartbeat**: the holder rewrites `heartbeat` every 60 s while it works,
  and checks the token before every write of `index.bin`. If the token is no
  longer its own, it stops: another machine took over.
- **Stale**: a lock whose heartbeat is older than 15 minutes, or whose `host`
  is this machine and whose `pid` no longer runs, may be taken over: delete it,
  then acquire as above.
- **Release**: delete the file if the token is still ours.
- Users can break a lock by hand (the file is just deleted).

## index.bin

All integers little-endian. Strings are UTF-8, length-prefixed, not
NUL-terminated.

```
magic         8 bytes   "SEMINDEX"
version       u32       1
headerLength  u32
header        bytes     UTF-8 JSON, see below
itemCount     u32
items         itemCount × item
```

Header JSON:

```json
{"model": "embeddinggemma-2", "quant": "UD-Q4_K_XL", "dim": 256, "prompts": 1,
 "chunk": [1200, 150, 8], "lastIndexedAt": 1793001234567,
 "writer": "krunner-embed 0.2.0 on ege-Desktop"}
```

`lastIndexedAt` is the end of the last *completed* pass (ms since the Unix
epoch, UTC), or 0.

Item:

```
pathLength    u16
path          bytes      relative to the index root, "/"-separated, no leading "/"
kind          u8         0 document, 1 image, 2 video, 3 audio
flags         u8         0 (reserved)
mtime         i64        ms since the Unix epoch, UTC
size          i64        bytes
hashLength    u8
hash          bytes      documents: SHA-1 of (file name, NUL, extracted text); text
                         extraction differs between apps, so only the writer's own
                         hashes are sure to match
                         media: SHA-1 of (size as i64 LE, first 64 KiB, last 64 KiB)
rows          u16        0 = the file could not be indexed (skipped until it changes)
scales        rows × f32
vectors       rows × 256 × i8
previews      rows × (u16 length, bytes)   start of each passage, ≤160 characters; empty for media
```

## Matching files across machines

- **Paths**: compare in Unicode NFC. macOS may report decomposed (NFD) names,
  Linux reports the bytes on disk. Keep the name as found for opening the file.
- **Unchanged file**: same size, and mtime within 2 s, or a whole number of
  hours (up to 14) ± 2 s apart. FAT stores local time with 2 s precision, so
  a time-zone or DST change shifts every timestamp. When in doubt, compare the
  hash before re-embedding.
