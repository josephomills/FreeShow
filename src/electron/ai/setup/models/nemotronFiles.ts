// AI AUTO SCRIPTURE - the pinned Nemotron model set, as data only.
//
// Split out of nemotron.ts so the file names and the source revision can be read WITHOUT pulling
// in the downloader (which imports electron's net/IPC). The runtime loader
// (speech/nemotron/manager.ts) and the benchmark harness (speech/bench/) both need the names but
// only one of them runs inside electron.
//
// int8 export of NVIDIA's Nemotron 3.5 streaming transducer (multilingual), converted for
// sherpa-onnx. Chosen over the English-only 0.6b on measured evidence over 19 sermon excerpts
// (src/electron/ai/speech/bench/): identical CPU cost, worst-case word latency 1400ms against
// 2500ms, the best accuracy of the five variants tried, and the fewest visibly repeated words -
// while also covering ~40 languages and emitting punctuation and capitalisation of its own.
//
// The chunk grid is 1120ms, the same as the export it replaces, so NEMOTRON_CHUNK_SHIFT_MS below
// and every commit timing derived from it are unchanged. Lower-latency exports of these weights
// exist at 80/160/320/560ms and measured WORSE: 2-4x the CPU and 6-7 points less accurate, because
// a shorter chunk gives the encoder less context to work with.
// Pinned to a specific repo revision (not "main") and to per-file SHA-256 hashes, so exactly
// these bytes land or nothing does - the hashes are the LFS checksums Hugging Face publishes for
// this revision.
/**
 * The pinned repo revision. Stamped beside a verified model (speech/nemotron/integrity.ts) so a
 * model left over from an earlier pin is recognised as outdated instead of being loaded silently.
 * Must be changed together with MODEL_BASE_URL and the hashes below.
 */
export const NEMOTRON_MODEL_REVISION = "cba1c96ca5ef0e8393b50584ae153a79145dc492"

export const MODEL_BASE_URL = "https://huggingface.co/csukuangfj2/sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-1120ms-int8-2026-06-11/resolve/cba1c96ca5ef0e8393b50584ae153a79145dc492"

export const NEMOTRON_MODEL_FILES = {
    encoder: { file: "encoder.int8.onnx", sha256: "2fff2166acaa535bd969fb223c1f0783d71029f143cb298bc54c2afe85abf772" },
    decoder: { file: "decoder.int8.onnx", sha256: "19f9c98fc6d0a2c33a65a43b36fdb2e914c26c0aa9764be3aebc502a1e982fb0" },
    joiner: { file: "joiner.int8.onnx", sha256: "4101c7c679a0bc30483794b27a059e34e79232aa2068d78d51231a22c8b0d7ce" },
    tokens: { file: "tokens.txt", sha256: "729cc103155bafa785f9cd45746cd41cabe97eab7182fc04d594129587958f8a" }
}
export const NEMOTRON_MODEL_BYTES = 682_200_000

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
