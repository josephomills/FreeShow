// AI BENCH - the model sets a run can be pointed at.
//
// The shipped engine pins ONE sherpa-onnx export (setup/models/nemotronFiles.ts). Two things
// about that pin are worth questioning and neither can be argued without measuring both sides:
//
//  - CHUNK TIER. The encoder's chunk grid is baked into the export, and no runtime setting moves
//    it, so it is the engine's hard latency floor. We ship the 1120ms build; the same NVIDIA
//    weights are published at 80/160/560ms. Smaller means lower latency and less context per
//    step, and NVIDIA's own figures show accuracy degrading as it shrinks - so where the knee is
//    is an empirical question about THIS audio, not a general one.
//  - MODEL. The English-only 0.6b cannot serve FreeShow's 30 non-English locales, which is why
//    resolveSttEngine() sends them to whisper. Nemotron 3.5 covers ~40 languages at the same
//    size. Whether that costs English accuracy is, again, measurable.
//
// Sets are downloaded into bin/bench/models/<id>/ by scripts/ai/getBenchModels.js and are never
// part of the app - the shipped model stays where the engine expects it.

export interface BenchModelSet {
    id: string
    /** Hugging Face repo, pinned to a revision by the downloader. */
    repo: string
    /** Chunk grid of this export, in ms. The latency floor. */
    chunkMs: number
    languages: "en" | "multi"
    /** Present only where the export ships one - required for hotwords/contextual biasing. */
    hasBpeModel: boolean
    notes?: string
}

export const BENCH_MODEL_SETS: BenchModelSet[] = [
    {
        id: "en-1120",
        repo: "csukuangfj/sherpa-onnx-nemotron-speech-streaming-en-0.6b-int8-2026-01-14",
        chunkMs: 1120,
        languages: "en",
        hasBpeModel: false,
        notes: "what FreeShow ships today - the baseline every other row is compared against"
    },
    {
        id: "en-160",
        repo: "csukuangfj2/sherpa-onnx-nemotron-speech-streaming-en-0.6b-160ms-int8-2026-04-25",
        chunkMs: 160,
        languages: "en",
        hasBpeModel: false,
        notes: "same weights as en-1120, 7x finer grid - isolates what the chunk tier costs"
    },
    {
        id: "multi-1120",
        repo: "csukuangfj2/sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-1120ms-int8-2026-06-11",
        chunkMs: 1120,
        languages: "multi",
        hasBpeModel: false,
        notes: "same tier as en-1120 - isolates what multilingual costs on English"
    },
    {
        id: "multi-320",
        repo: "csukuangfj2/sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-320ms-int8-2026-06-11",
        chunkMs: 320,
        languages: "multi",
        hasBpeModel: false,
        notes: "the candidate: multilingual at a low-latency tier"
    }
]

/**
 * Parked, not dismissed. The Zipformer is the ONLY cross-platform route to contextual biasing -
 * hotwords need modified_beam_search plus a BPE vocabulary, and no Nemotron export supports either
 * (k2-fsa/sherpa-onnx#3572). Its other numbers are striking: 72 MB against Nemotron's 660 MB, and
 * it decodes at RTF 0.024 against the streaming Nemotron's 0.06.
 *
 * Two things stopped it here. It transcribes garbage under every file pairing tried so far - int8
 * and fp32 encoder, int8 and fp32 decoder - producing plausible but wrong English ("UNCLE YELLOW"
 * for "AFTER EARLY NIGHTFALL THE YELLOW LAMPS"), on LibriSpeech as well as on sermon audio, so it
 * is a configuration fault rather than a domain mismatch. And the repo ships bpe.model but not
 * bpe.vocab, which is what sherpa's hotword path actually wants ("Each line in vocab should contain
 * two items ... the first one is bpe token, the second one is score") - exporting it needs
 * sentencepiece.
 *
 * Worth returning to once the tier and multilingual questions are settled.
 */
export const ZIPFORMER_PARKED = true

export function findModelSet(id: string): BenchModelSet | undefined {
    return BENCH_MODEL_SETS.find((set) => set.id === id)
}
