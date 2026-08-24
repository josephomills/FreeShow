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
 * Parked after a real attempt, not dismissed. It remains the most interesting model here on every
 * axis except the one that matters: 72 MB against Nemotron's 682 MB, 0.27 GB of RAM against 2.0 GB,
 * RTF 0.024 against 0.06 - and it is the ONLY cross-platform route to contextual biasing, because
 * hotwords need modified_beam_search plus a BPE vocabulary and no Nemotron export supports either
 * (k2-fsa/sherpa-onnx#3572).
 *
 * It transcribes plausible-but-wrong English. On the LibriSpeech clip the repo itself ships, whose
 * reference is "AFTER EARLY NIGHTFALL THE YELLOW LAMPS...", it returns "YOU LIKE A MAN OF THE" and
 * "UNCLE ELD ME OH". Coherent words, correct casing, drawn from the right vocabulary - the shape of
 * a feature mismatch, not of a corrupt file or a domain gap.
 *
 * Ruled out, each tested against that known-good clip:
 *   - quantization: int8 and fp32 encoder, int8 and fp32 decoder and joiner, every combination
 *   - streaming context: both the left-128 and left-64 exports
 *   - flush: inputFinished() as well as trailing silence
 *   - feed shape: 100ms chunks and the whole clip in one call
 *   - sample scaling: [-1,1] is correct (int16 range returns nothing at all)
 *   - feature options: dither, snipEdges
 *   - the token table: 502 entries, uppercase, and the output words come from it, so the id-to-text
 *     mapping is right and the acoustic model is simply choosing wrong ids
 *
 * The untested lead, and the right next step: run this model through sherpa-onnx's own CLI. If the
 * CLI transcribes it correctly then the fault is in the Node binding's feature path and none of the
 * above would ever have found it; if the CLI fails too, the export is at fault and belongs upstream.
 *
 * Its hotword path needs a second thing regardless: the repo ships bpe.model, while sherpa wants
 * bpe.vocab ("Each line in vocab should contain two items ... the first one is bpe token, the second
 * one is score"), which requires sentencepiece to export.
 */
export const ZIPFORMER_PARKED = true

export function findModelSet(id: string): BenchModelSet | undefined {
    return BENCH_MODEL_SETS.find((set) => set.id === id)
}
