// AI AUTO SCRIPTURE - the pinned Nemotron model set, as data only.
//
// Split out of nemotron.ts so the file names and the source revision can be read WITHOUT pulling
// in the downloader (which imports electron's net/IPC). The runtime loader
// (speech/nemotron/manager.ts) and the benchmark harness (speech/bench/) both need the names but
// only one of them runs inside electron.
//
// int8 export of NVIDIA's streaming Nemotron transducer, converted for sherpa-onnx.
// Pinned to a specific repo revision (not "main") and to per-file SHA-256 hashes, so exactly
// these bytes land or nothing does - the hashes are the LFS checksums Hugging Face publishes for
// this revision.
/**
 * The pinned repo revision. Stamped beside a verified model (speech/nemotron/integrity.ts) so a
 * model left over from an earlier pin is recognised as outdated instead of being loaded silently.
 * Must be changed together with MODEL_BASE_URL and the hashes below.
 */
export const NEMOTRON_MODEL_REVISION = "f13b0c6a48186fdd9fdd8d203b9527b0b709b09f"

export const MODEL_BASE_URL = "https://huggingface.co/csukuangfj/sherpa-onnx-nemotron-speech-streaming-en-0.6b-int8-2026-01-14/resolve/f13b0c6a48186fdd9fdd8d203b9527b0b709b09f"

export const NEMOTRON_MODEL_FILES = {
    encoder: { file: "encoder.int8.onnx", sha256: "2f6ae81fe4ccd69ef04cdf048ecd49628e2d3148a6195e152a91b4d2497952dc" },
    decoder: { file: "decoder.int8.onnx", sha256: "1fb1795cb46e7d0e99b2e096eae83f7e324294e895975a1a894b0384cbbe37f6" },
    joiner: { file: "joiner.int8.onnx", sha256: "a3f41dccc0f67f37e4210051d1c39a29d473c841cfc32fe574135bac890db91d" },
    tokens: { file: "tokens.txt", sha256: "dc0b4584ab2e4ddbf888425c076c61b736e7356a015250db7d307e6f1a8188ff" }
}
export const NEMOTRON_MODEL_BYTES = 661_920_000

// speech gating, shared by any streaming driver (~630 KB)
export const VAD_MODEL_URL = "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx"
export const NEMOTRON_VAD_FILE = "silero_vad.onnx"
export const VAD_MODEL_SHA256 = "9e2449e1087496d8d4caba907f23e0bd3f78d91fa552479bb9c23ac09cbb1fd6"

/**
 * The encoder's chunk grid, read from the ONNX metadata (window_size=121, chunk_shift=112 frames
 * at 10 ms per frame). It is baked into the export and no runtime setting moves it: the first
 * encoder step needs PRIMING_MS of buffered audio, every step after it fires CHUNK_SHIFT_MS
 * later. Together they are the engine's latency floor, and the reason a fresh stream per decode
 * costs so much - each one re-pays the priming.
 *
 * Lower-latency exports of the same NVIDIA weights are published at 80/160/560 ms; swapping to
 * one is a change to MODEL_BASE_URL and the four hashes above, and these two numbers.
 */
export const NEMOTRON_CHUNK_SHIFT_MS = 1120
export const NEMOTRON_PRIMING_MS = 1300
