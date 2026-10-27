import AVFoundation
import AppKit
import Foundation
import ImageIO
import PDFKit
import llama

// SuperCmd semantic search helper.
//
// Runs EmbeddingGemma 2 through llama.cpp (Metal) and speaks JSON lines on
// stdin/stdout, like whisper-transcriber's serve mode:
//
//   embedding-server serve --model <text.gguf> [--mmproj <mmproj.gguf>]
//
// Commands (every request may carry an "id", echoed back in the response):
//   {"command":"embed","texts":[...],"dim":256}            -> {"embeddings":"<base64 f32>","count":n,"dim":256}
//   {"command":"embed_media","path":"...","kind":"image","dim":256} -> {"embedding":"<base64 f32>","dim":256}
//   {"command":"extract_text","path":"...","max_chars":20000} -> {"text":"..."}
//   {"command":"release_media"} / {"command":"ping"} / {"command":"exit"}
//
// The vision/audio encoder (mmproj) is loaded lazily on the first media
// request and released after a short idle period, so text-only search keeps
// the footprint down to the text model.

enum EmbeddingServerError: Error, CustomStringConvertible {
  case failed(String)

  var description: String {
    switch self {
    case .failed(let message):
      return message
    }
  }
}

private let contextTokens: UInt32 = 4096
private let maxSequencesPerBatch: UInt32 = 16
private let maxTokensPerText = 2048
private let maxVideoFrames = 8
private let maxAudioSeconds = 60.0
private let imageMaxPixelSize = 896
private let mediaIdleReleaseSeconds: TimeInterval = 90

// MARK: - JSON helpers

func emitJSON(_ dict: [String: Any]) {
  guard let data = try? JSONSerialization.data(withJSONObject: dict),
        let line = String(data: data, encoding: .utf8)
  else { return }
  FileHandle.standardOutput.write(Data((line + "\n").utf8))
}

func base64Floats(_ values: [Float]) -> String {
  values.withUnsafeBufferPointer { Data(buffer: $0) }.base64EncodedString()
}

/// Matryoshka truncation: keep the leading `dim` values and re-normalize.
func truncateAndNormalize(_ values: UnsafePointer<Float>, count: Int, dim: Int) -> [Float] {
  let n = (dim > 0 && dim < count) ? dim : count
  var out = Array(UnsafeBufferPointer(start: values, count: n))
  var norm: Float = 0
  for v in out { norm += v * v }
  norm = norm.squareRoot()
  if norm > 0 {
    for i in 0..<n { out[i] /= norm }
  }
  return out
}

// MARK: - Engine

final class EmbeddingEngine {
  let model: OpaquePointer
  let context: OpaquePointer
  let vocab: OpaquePointer
  let batch: OpaquePointer
  let nEmbdOut: Int
  let nEmbdInp: Int
  let mmprojPath: String?
  private var media: OpaquePointer?
  private var mediaLastUsed = Date.distantPast

  init(modelPath: String, mmprojPath: String?) throws {
    llama_backend_init()

    var modelParams = llama_model_default_params()
    modelParams.n_gpu_layers = -1
    guard let model = llama_model_load_from_file(modelPath, modelParams) else {
      throw EmbeddingServerError.failed("Failed to load embedding model at \(modelPath)")
    }

    var ctxParams = llama_context_default_params()
    ctxParams.n_ctx = contextTokens
    ctxParams.n_batch = contextTokens
    ctxParams.n_ubatch = contextTokens
    ctxParams.n_seq_max = maxSequencesPerBatch
    ctxParams.embeddings = true
    ctxParams.pooling_type = LLAMA_POOLING_TYPE_MEAN
    ctxParams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_AUTO
    let threads = Int32(max(2, min(4, ProcessInfo.processInfo.activeProcessorCount / 2)))
    ctxParams.n_threads = threads
    ctxParams.n_threads_batch = threads

    guard let context = llama_init_from_model(model, ctxParams) else {
      llama_model_free(model)
      throw EmbeddingServerError.failed("Failed to create llama.cpp context")
    }
    guard let batch = llama_batch_ext_init(context) else {
      llama_free(context)
      llama_model_free(model)
      throw EmbeddingServerError.failed("Failed to allocate llama.cpp batch")
    }

    self.model = model
    self.context = context
    self.vocab = llama_model_get_vocab(model)
    self.batch = batch
    self.nEmbdOut = Int(llama_model_n_embd_out(model))
    self.nEmbdInp = Int(llama_model_n_embd_inp(model))
    self.mmprojPath = (mmprojPath?.isEmpty ?? true) ? nil : mmprojPath
  }

  deinit {
    releaseMedia()
    llama_batch_ext_free(batch)
    llama_free(context)
    llama_model_free(model)
    llama_backend_free()
  }

  // MARK: Text

  func tokenize(_ text: String) -> [llama_token] {
    // parse_special matches the Hugging Face tokenizer, which matches added
    // tokens inside plain text (e.g. markup in source files).
    let utf8Count = text.utf8.count
    var tokens = [llama_token](repeating: 0, count: utf8Count + 8)
    var n = llama_tokenize(vocab, text, Int32(utf8Count), &tokens, Int32(tokens.count), true, true)
    if n < 0 {
      tokens = [llama_token](repeating: 0, count: Int(-n))
      n = llama_tokenize(vocab, text, Int32(utf8Count), &tokens, Int32(tokens.count), true, true)
    }
    tokens = Array(tokens.prefix(Int(max(0, n))))
    if tokens.count > maxTokensPerText {
      // Keep the trailing special token (EOS) when the vocab appends one.
      let last = tokens.last!
      tokens = Array(tokens.prefix(maxTokensPerText))
      if llama_vocab_get_add_eos(vocab) { tokens[tokens.count - 1] = last }
    }
    return tokens
  }

  func embedTexts(_ texts: [String], dim: Int) throws -> [[Float]] {
    var results = [[Float]](repeating: [], count: texts.count)
    var pending: [(index: Int, tokens: [llama_token])] = []
    var pendingTokens = 0

    func flush() throws {
      if pending.isEmpty { return }
      clearMemory()
      llama_batch_ext_clear(batch)
      for (seq, entry) in pending.enumerated() {
        for (pos, token) in entry.tokens.enumerated() {
          let idx = llama_batch_ext_add_token(batch, llama_seq_id(seq), token)
          if idx < 0 { throw EmbeddingServerError.failed("Batch is full (error \(idx))") }
          var positions: [llama_pos] = [llama_pos(pos), 0, 0, 0]
          _ = llama_batch_ext_set_pos(batch, idx, &positions)
          _ = llama_batch_ext_set_output_embd(batch, idx, true)
        }
      }
      let rc = llama_process(context, LLAMA_PROCESS_TYPE_DECODE, batch)
      if rc < 0 { throw EmbeddingServerError.failed("llama_process failed (\(rc))") }
      for (seq, entry) in pending.enumerated() {
        guard let embd = llama_get_embeddings_seq(context, llama_seq_id(seq)) else {
          throw EmbeddingServerError.failed("No embedding produced for sequence \(seq)")
        }
        results[entry.index] = truncateAndNormalize(embd, count: nEmbdOut, dim: dim)
      }
      pending.removeAll(keepingCapacity: true)
      pendingTokens = 0
    }

    for (index, text) in texts.enumerated() {
      let tokens = tokenize(text)
      if tokens.isEmpty {
        results[index] = [Float](repeating: 0, count: (dim > 0 && dim < nEmbdOut) ? dim : nEmbdOut)
        continue
      }
      if pending.count >= Int(maxSequencesPerBatch) || pendingTokens + tokens.count > Int(contextTokens) {
        try flush()
      }
      pending.append((index, tokens))
      pendingTokens += tokens.count
    }
    try flush()
    return results
  }

  private func clearMemory() {
    if let memory = llama_get_memory(context) {
      llama_memory_clear(memory, true)
    }
  }

  // MARK: Media

  private func ensureMedia() throws -> OpaquePointer {
    if let media {
      mediaLastUsed = Date()
      return media
    }
    guard let path = mmprojPath else {
      throw EmbeddingServerError.failed("Image, video and audio embedding needs the media encoder (mmproj) model")
    }
    var params = mtmd_context_params_default()
    params.use_gpu = true
    params.print_timings = false
    params.warmup = false
    params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_AUTO
    params.n_threads = Int32(max(2, min(4, ProcessInfo.processInfo.activeProcessorCount / 2)))
    guard let ctx = mtmd_init_from_file(path, model, params) else {
      throw EmbeddingServerError.failed("Failed to load media encoder at \(path)")
    }
    media = ctx
    mediaLastUsed = Date()
    return ctx
  }

  func releaseMedia() {
    if let media {
      mtmd_free(media)
      self.media = nil
    }
  }

  func releaseMediaIfIdle() {
    if media != nil && Date().timeIntervalSince(mediaLastUsed) > mediaIdleReleaseSeconds {
      releaseMedia()
    }
  }

  func embedMedia(path: String, kind: String, dim: Int) throws -> [Float] {
    let mctx = try ensureMedia()
    defer { mediaLastUsed = Date() }

    var bitmaps: [OpaquePointer] = []
    defer { bitmaps.forEach { mtmd_bitmap_free($0) } }

    switch kind {
    case "image":
      guard mtmd_support_vision(mctx) else { throw EmbeddingServerError.failed("Media encoder has no vision support") }
      bitmaps.append(try rgbBitmap(from: try loadImage(path: path)))
    case "video":
      guard mtmd_support_vision(mctx) else { throw EmbeddingServerError.failed("Media encoder has no vision support") }
      for frame in try loadVideoFrames(path: path) {
        bitmaps.append(try rgbBitmap(from: frame))
      }
    case "audio":
      guard mtmd_support_audio(mctx) else { throw EmbeddingServerError.failed("Media encoder has no audio support") }
      let rate = Double(mtmd_get_audio_sample_rate(mctx))
      let samples = try loadAudio(path: path, sampleRate: rate > 0 ? rate : 16_000)
      guard let bitmap = samples.withUnsafeBufferPointer({ mtmd_bitmap_init_from_audio($0.count, $0.baseAddress) }) else {
        throw EmbeddingServerError.failed("Failed to wrap audio samples")
      }
      bitmaps.append(bitmap)
    default:
      throw EmbeddingServerError.failed("Unknown media kind: \(kind)")
    }
    if bitmaps.isEmpty { throw EmbeddingServerError.failed("No decodable media in \(path)") }

    // Media goes in without a task prefix (per the model card): just one
    // marker per bitmap.
    let marker = String(cString: mtmd_default_marker())
    let prompt = Array(repeating: marker, count: bitmaps.count).joined(separator: " ")
    guard let chunks = mtmd_input_chunks_init() else { throw EmbeddingServerError.failed("mtmd_input_chunks_init failed") }
    defer { mtmd_input_chunks_free(chunks) }

    let tokenizeResult: Int32 = prompt.withCString { cPrompt in
      var input = mtmd_input_text(text: cPrompt, text_len: strlen(cPrompt), add_special: true, parse_special: true)
      var bitmapPointers: [OpaquePointer?] = bitmaps.map { Optional($0) }
      return bitmapPointers.withUnsafeMutableBufferPointer { buffer in
        mtmd_tokenize(mctx, chunks, &input, buffer.baseAddress, buffer.count)
      }
    }
    if tokenizeResult != 0 { throw EmbeddingServerError.failed("mtmd_tokenize failed (\(tokenizeResult))") }

    // Encode every media chunk first and keep the rows in one buffer, so the
    // batch below can point into storage that no longer moves.
    enum Entry {
      case token(llama_token, [llama_pos])
      case row(Int, [llama_pos])
    }
    var entries: [Entry] = []
    var rows: [Float] = []
    let useMrope = mtmd_decode_use_mrope(mctx)
    var nPast: llama_pos = 0

    for i in 0..<mtmd_input_chunks_size(chunks) {
      guard let chunk = mtmd_input_chunks_get(chunks, i) else { continue }
      if mtmd_input_chunk_get_type(chunk) == MTMD_INPUT_CHUNK_TYPE_TEXT {
        var nTokens = 0
        guard let tokens = mtmd_input_chunk_get_tokens_text(chunk, &nTokens) else { continue }
        for t in 0..<nTokens {
          entries.append(.token(tokens[t], [nPast, 0, 0, 0]))
          nPast += 1
        }
        continue
      }

      let rc = mtmd_encode_chunk(mctx, chunk)
      if rc != 0 { throw EmbeddingServerError.failed("Media encode failed (\(rc))") }
      guard let output = mtmd_get_output_embd(mctx) else { throw EmbeddingServerError.failed("Media encoder produced no output") }
      let nTokens = Int(mtmd_input_chunk_get_n_tokens(chunk))
      let image = mtmd_input_chunk_get_tokens_image(chunk)
      for t in 0..<nTokens {
        var positions: [llama_pos] = [nPast + llama_pos(t), nPast + llama_pos(t), nPast + llama_pos(t), 0]
        if useMrope, let image {
          let rel = mtmd_image_tokens_get_decoder_pos(image, nPast, t)
          positions = [llama_pos(rel.t), llama_pos(rel.y), llama_pos(rel.x), llama_pos(rel.z)]
        }
        entries.append(.row(rows.count / nEmbdInp, positions))
        rows.append(contentsOf: UnsafeBufferPointer(start: output + t * nEmbdInp, count: nEmbdInp))
      }
      nPast += mtmd_input_chunk_get_n_pos(chunk)
    }

    if entries.count > Int(contextTokens) {
      throw EmbeddingServerError.failed("Media input needs \(entries.count) tokens, limit is \(contextTokens)")
    }

    clearMemory()
    llama_batch_ext_clear(batch)
    try rows.withUnsafeBufferPointer { rowBuffer in
      for entry in entries {
        switch entry {
        case .token(let token, var positions):
          let idx = llama_batch_ext_add_token(batch, 0, token)
          if idx < 0 { throw EmbeddingServerError.failed("Batch is full (error \(idx))") }
          _ = llama_batch_ext_set_pos(batch, idx, &positions)
          _ = llama_batch_ext_set_output_embd(batch, idx, true)
        case .row(let row, var positions):
          let embd = llama_embd(data: rowBuffer.baseAddress! + row * nEmbdInp, n_rows: 1, n_embd: nEmbdInp)
          let idx = llama_batch_ext_add_embd(batch, 0, embd)
          if idx < 0 { throw EmbeddingServerError.failed("Batch is full (error \(idx))") }
          _ = llama_batch_ext_set_pos(batch, idx, &positions)
          _ = llama_batch_ext_set_output_embd(batch, idx, true)
        }
      }
      let rc = llama_process(context, LLAMA_PROCESS_TYPE_DECODE, batch)
      if rc < 0 { throw EmbeddingServerError.failed("llama_process failed (\(rc))") }
    }

    guard let embd = llama_get_embeddings_seq(context, 0) else {
      throw EmbeddingServerError.failed("No embedding produced for media")
    }
    return truncateAndNormalize(embd, count: nEmbdOut, dim: dim)
  }
}

// MARK: - Media decoding

func loadImage(path: String) throws -> CGImage {
  let url = URL(fileURLWithPath: path) as CFURL
  guard let source = CGImageSourceCreateWithURL(url, nil) else {
    throw EmbeddingServerError.failed("Unreadable image: \(path)")
  }
  // Decode straight to a bounded size; the encoder resizes anyway and this
  // keeps 48MP photos from ballooning memory.
  let options: [CFString: Any] = [
    kCGImageSourceCreateThumbnailFromImageAlways: true,
    kCGImageSourceCreateThumbnailWithTransform: true,
    kCGImageSourceThumbnailMaxPixelSize: imageMaxPixelSize,
  ]
  guard let image = CGImageSourceCreateThumbnailAtIndex(source, 0, options as CFDictionary) else {
    throw EmbeddingServerError.failed("Could not decode image: \(path)")
  }
  return image
}

func rgbBitmap(from image: CGImage) throws -> OpaquePointer {
  let width = image.width
  let height = image.height
  var rgba = [UInt8](repeating: 0, count: width * height * 4)
  let drawn: Bool = rgba.withUnsafeMutableBytes { buffer in
    guard let ctx = CGContext(
      data: buffer.baseAddress,
      width: width,
      height: height,
      bitsPerComponent: 8,
      bytesPerRow: width * 4,
      space: CGColorSpaceCreateDeviceRGB(),
      bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue
    ) else { return false }
    ctx.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
    return true
  }
  if !drawn { throw EmbeddingServerError.failed("Could not rasterize image") }

  var rgb = [UInt8](repeating: 0, count: width * height * 3)
  for p in 0..<(width * height) {
    rgb[p * 3] = rgba[p * 4]
    rgb[p * 3 + 1] = rgba[p * 4 + 1]
    rgb[p * 3 + 2] = rgba[p * 4 + 2]
  }
  guard let bitmap = rgb.withUnsafeBufferPointer({ mtmd_bitmap_init(UInt32(width), UInt32(height), $0.baseAddress) }) else {
    throw EmbeddingServerError.failed("Failed to wrap image")
  }
  return bitmap
}

final class BlockingResult<T>: @unchecked Sendable {
  var value: Result<T, Error>?
}

/// Run an async API synchronously; the helper is a single-threaded CLI loop.
func blocking<T>(_ operation: @escaping @Sendable () async throws -> T) throws -> T {
  let semaphore = DispatchSemaphore(value: 0)
  let box = BlockingResult<T>()
  Task {
    do { box.value = .success(try await operation()) } catch { box.value = .failure(error) }
    semaphore.signal()
  }
  semaphore.wait()
  return try box.value!.get()
}

/// Samples up to `maxVideoFrames` frames, one per second (the model's video
/// default) or spread evenly across longer clips.
func loadVideoFrames(path: String) throws -> [CGImage] {
  let asset = AVURLAsset(url: URL(fileURLWithPath: path))
  let duration = try blocking { try await asset.load(.duration) }.seconds
  guard duration.isFinite, duration > 0 else { throw EmbeddingServerError.failed("Video has no duration: \(path)") }

  let count = max(1, min(maxVideoFrames, Int(duration.rounded(.down))))
  let step = duration / Double(count)
  let times = (0..<count).map { CMTime(seconds: step * (Double($0) + 0.5), preferredTimescale: 600) }

  let generator = AVAssetImageGenerator(asset: asset)
  generator.appliesPreferredTrackTransform = true
  generator.maximumSize = CGSize(width: imageMaxPixelSize, height: imageMaxPixelSize)
  generator.requestedTimeToleranceBefore = CMTime(seconds: 0.5, preferredTimescale: 600)
  generator.requestedTimeToleranceAfter = CMTime(seconds: 0.5, preferredTimescale: 600)

  var frames: [CGImage] = []
  for time in times {
    if let frame = try? blocking({ try await generator.image(at: time).image }) {
      frames.append(frame)
    }
  }
  if frames.isEmpty { throw EmbeddingServerError.failed("Could not read video frames: \(path)") }
  return frames
}

/// Decodes any AVFoundation-readable audio to mono float PCM at `sampleRate`,
/// capped at `maxAudioSeconds`.
func loadAudio(path: String, sampleRate: Double) throws -> [Float] {
  let file = try AVAudioFile(forReading: URL(fileURLWithPath: path))
  let inputFormat = file.processingFormat
  guard let outputFormat = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: sampleRate, channels: 1, interleaved: false),
        let converter = AVAudioConverter(from: inputFormat, to: outputFormat)
  else { throw EmbeddingServerError.failed("Unsupported audio format: \(path)") }

  let maxInputFrames = AVAudioFrameCount(min(Double(file.length), maxAudioSeconds * inputFormat.sampleRate))
  guard let input = AVAudioPCMBuffer(pcmFormat: inputFormat, frameCapacity: maxInputFrames) else {
    throw EmbeddingServerError.failed("Could not allocate audio buffer")
  }
  try file.read(into: input, frameCount: maxInputFrames)

  let outputCapacity = AVAudioFrameCount(Double(input.frameLength) * sampleRate / inputFormat.sampleRate) + 1024
  guard let output = AVAudioPCMBuffer(pcmFormat: outputFormat, frameCapacity: outputCapacity) else {
    throw EmbeddingServerError.failed("Could not allocate audio buffer")
  }
  var consumed = false
  var conversionError: NSError?
  converter.convert(to: output, error: &conversionError) { _, status in
    if consumed {
      status.pointee = .endOfStream
      return nil
    }
    consumed = true
    status.pointee = .haveData
    return input
  }
  if let conversionError { throw conversionError }
  guard let channel = output.floatChannelData?[0], output.frameLength > 0 else {
    throw EmbeddingServerError.failed("Audio decoded to no samples: \(path)")
  }
  return Array(UnsafeBufferPointer(start: channel, count: Int(output.frameLength)))
}

// MARK: - Document text extraction

func extractText(path: String, maxChars: Int) throws -> String {
  let url = URL(fileURLWithPath: path)
  var text = ""
  if url.pathExtension.lowercased() == "pdf" {
    guard let document = PDFDocument(url: url) else { throw EmbeddingServerError.failed("Unreadable PDF: \(path)") }
    for index in 0..<document.pageCount {
      if let page = document.page(at: index)?.string {
        text += page + "\n"
      }
      if text.count >= maxChars { break }
    }
  } else {
    // rtf, rtfd, doc, docx, odt, html, webarchive, ...
    let attributed = try NSAttributedString(url: url, options: [:], documentAttributes: nil)
    text = attributed.string
  }
  return String(text.prefix(maxChars))
}

// MARK: - Persistent server mode

func serveCommand(modelPath: String, mmprojPath: String?) {
  llama_log_set({ level, text, _ in
    guard level == GGML_LOG_LEVEL_ERROR, let text else { return }
    FileHandle.standardError.write(Data(String(cString: text).utf8))
  }, nil)
  mtmd_log_set({ level, text, _ in
    guard level == GGML_LOG_LEVEL_ERROR, let text else { return }
    FileHandle.standardError.write(Data(String(cString: text).utf8))
  }, nil)

  let engine: EmbeddingEngine
  do {
    engine = try EmbeddingEngine(modelPath: modelPath, mmprojPath: mmprojPath)
  } catch {
    emitJSON(["error": String(describing: error)])
    return
  }

  emitJSON(["ready": true, "dim": engine.nEmbdOut, "media": engine.mmprojPath != nil])

  while let line = readLine(strippingNewline: true) {
    let trimmed = line.trimmingCharacters(in: .whitespacesAndNewlines)
    if trimmed.isEmpty { continue }

    guard let data = trimmed.data(using: .utf8),
          let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
    else {
      emitJSON(["error": "Invalid JSON request"])
      continue
    }

    let id = json["id"] ?? NSNull()
    let cmd = json["command"] as? String ?? ""
    let dim = json["dim"] as? Int ?? 0

    do {
      switch cmd {
      case "embed":
        let texts = json["texts"] as? [String] ?? []
        let vectors = try engine.embedTexts(texts, dim: dim)
        emitJSON(["id": id, "embeddings": base64Floats(vectors.flatMap { $0 }), "count": vectors.count, "dim": vectors.first?.count ?? 0])
      case "embed_media":
        guard let path = json["path"] as? String, FileManager.default.fileExists(atPath: path) else {
          throw EmbeddingServerError.failed("Media file not found")
        }
        let vector = try engine.embedMedia(path: path, kind: json["kind"] as? String ?? "image", dim: dim)
        emitJSON(["id": id, "embedding": base64Floats(vector), "dim": vector.count])
      case "extract_text":
        guard let path = json["path"] as? String else { throw EmbeddingServerError.failed("Missing 'path' field") }
        let text = try extractText(path: path, maxChars: json["max_chars"] as? Int ?? 20_000)
        emitJSON(["id": id, "text": text])
      case "release_media":
        engine.releaseMedia()
        emitJSON(["id": id, "ok": true])
      case "ping":
        emitJSON(["id": id, "pong": true])
      case "exit":
        return
      default:
        throw EmbeddingServerError.failed("Unknown command: \(cmd)")
      }
    } catch {
      emitJSON(["id": id, "error": String(describing: error)])
    }
    engine.releaseMediaIfIdle()
  }
}

// MARK: - Main entry point

let args = Array(CommandLine.arguments.dropFirst())
guard args.first == "serve" else {
  FileHandle.standardError.write(Data("usage: embedding-server serve --model <gguf> [--mmproj <gguf>]\n".utf8))
  exit(1)
}

var modelPath = ""
var mmprojPath: String?
var argIndex = 1
while argIndex < args.count {
  let flag = args[argIndex]
  argIndex += 1
  guard argIndex < args.count else { break }
  if flag == "--model" || flag == "-m" {
    modelPath = args[argIndex]
  } else if flag == "--mmproj" {
    mmprojPath = args[argIndex]
  }
  argIndex += 1
}

guard !modelPath.isEmpty else {
  FileHandle.standardError.write(Data("serve mode requires --model <path>\n".utf8))
  exit(1)
}

serveCommand(modelPath: modelPath, mmprojPath: mmprojPath)
