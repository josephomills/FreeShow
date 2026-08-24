// AI AUTO SCRIPTURE - streaming transcription over sherpa-onnx (NVIDIA Nemotron), cache-aware.
//
// The sibling driver.ts decodes each utterance in one batch on a FRESH stream, repeatedly. This
// one keeps a SINGLE stream open for the whole session and feeds it only the new audio. Same
// model, same VAD, same emitted segment shape - the difference is entirely in how the recognizer
// is driven, and it is large. Measured on 120s of real preaching (bench/, AI_BENCH=1):
//
//                        driver.ts (batch)   this file (streaming)
//   decode cost              0.21x realtime        0.059x realtime
//   per-push p99 / max         468 / 607 ms            66 / 74 ms
//
// Three properties of the model make this work, all measured rather than assumed:
//
// 1. The encoder runs on a fixed grid: the first step needs 1300ms of buffered audio, every step
//    after it fires 1120ms later (NEMOTRON_PRIMING_MS / NEMOTRON_CHUNK_SHIFT_MS, read from the
//    ONNX metadata). Feeding a whole utterance at once and feeding it in 100ms pushes produce the
//    IDENTICAL number of encoder steps - OnlineStream buffers internally. So batching bought
//    nothing, while paying the 1300ms priming again on every single decode.
//
// 2. `getResult(stream).text` on a persistent stream is monotonic - always a prefix-extension of
//    the previous value. Verified over 120s of continuous preaching without a single exception.
//    Greedy RNN-T appends tokens frame by frame and cannot retract them. This is what removes the
//    "two consecutive decodes must agree" rule: that rule was defending against revisions which
//    only existed BECAUSE a fresh stream re-decoded the audio from scratch. Remove the cause and
//    the defence is pure latency. `assertPrefix` below keeps a cheap guard in place anyway.
//
// 3. `reset(stream)` clears the hypothesis but KEEPS the encoder cache. Verified by decoding the
//    same audio on a reset stream and on a brand-new one: they produce different text, so the
//    reset stream is still carrying acoustic context. That is what lets an utterance boundary
//    clear the transcript without re-paying the priming cost.
//
// Consequently the whole compensating apparatus in driver.ts - partial re-decodes, the agreement
// rule and its backoff, seam stitching, soft splits, split overlap, preroll, the finalize pad,
// the buffered utterance - has nothing left to compensate for and is absent here.
//
// What this does NOT change: greedy RNN-T is monotonic, so a word decoded wrong stays wrong.
// There is no revision and no hotword biasing at any latency (sherpa's Nemotron transducer
// supports greedy_search only - k2-fsa/sherpa-onnx#3572). Misheard biblical vocabulary is still
// recovered downstream by scripture/detection/asrRepairs.ts and the quote matcher's phonetic
// layer. And the 1120ms grid is baked into the ONNX export: no setting here moves it, but
// lower-latency exports of the same NVIDIA weights exist (see setup/models/nemotronFiles.ts).

import { NEMOTRON_CHUNK_SHIFT_MS } from "../../setup/models/nemotronFiles"
import { findRepeatedTail } from "../repetition"
import type { DriverCallbacks, TranscriberSegment, TranscriptionDriver } from "../types"
import type { NemotronModelPaths } from "./manager"

const SAMPLE_RATE = 16000

// VAD tuning is carried over from driver.ts unchanged - it was tuned against live services and
// none of the reasoning depends on how the recognizer is driven. A trailing word softened by room
// acoustics can dip below the speech threshold, and the silence countdown then runs DURING the
// word; a low threshold keeps quiet word endings counted as speech.
const VAD_THRESHOLD = 0.3
const VAD_MIN_SILENCE = 0.8
const VAD_MIN_SPEECH = 0.15
// only a ceiling on how long one hypothesis string may grow - text streams out continuously, so
// unlike driver.ts this is not a decode-cost boundary and never slices a word out of a batch
const VAD_MAX_SPEECH = 30

/**
 * Every timing below is derived from the export's chunk shift, so they must move together when the
 * export does. The shipped model is 1120ms, but the same NVIDIA weights are published at
 * 80/160/560ms and the multilingual 3.5 model at 80-1120ms; hardcoding 1120 would make a 160ms
 * export wait seven times too long to commit anything and look far worse than it is.
 *
 * The native side reads the true value from the ONNX metadata (chunk_shift) and does not expose
 * it, so this is passed in rather than discovered - and NEMOTRON_CHUNK_SHIFT_MS stays the default
 * so production and the pinned export can never drift apart.
 */
/**
 * How much audio may pass before the decoder is reset, on top of the reset at every utterance
 * boundary. 0 means the boundary is the only reset, which is the default and is deliberate.
 *
 * Carrying the RNN-T predictor's state across utterance boundaries measurably improved
 * transcription on 120 s fixtures - two more references recovered out of twenty-five - so this
 * defaulted to keeping it. Live use over a full service showed why that was wrong: the predictor's
 * own output is its next input, and given a long enough unbroken run it locks into a cycle and
 * fills the transcript with one phrase. The utterance boundary was bounding that, and the benefit
 * measured on two-minute clips cannot be seen to compound the way the failure does.
 *
 * The lesson is about the measurement, not the parameter: a fixture set of short clips cannot show
 * a degeneration that needs minutes of continuous decoding to appear.
 */
const DEFAULT_RESET_INTERVAL_MS = 0

function gridTimings(chunkShiftMs: number) {
    return {
        // Real audio that must reach the recognizer after the VAD says speech ended, before the
        // utterance is closed and the stream reset. It has to cover a FULL chunk shift: the last
        // word's audio only influences the transcript once an encoder step consumes it. driver.ts
        // used a flat 500ms, which is less than one 1120ms step - under a persistent stream that
        // closes before the final word's own chunk has run. Waiting costs nothing here: the audio
        // is being decoded either way, there is no batch to assemble.
        closeDeferSamples: Math.ceil(((chunkShiftMs + 100) / 1000) * SAMPLE_RATE),

        // While an utterance is open the trailing word is held back (it may be a mid-emission BPE
        // fragment). If the hypothesis then stops growing - a pause too short for the VAD to close
        // on - that word would wait indefinitely, which the bench measured at 9s worst case. A full
        // encoder step with no new tokens means the decoder had its chance and produced nothing, so
        // the word is committed. Bounds tail latency at about two chunk shifts.
        staticTailSamples: Math.ceil(((chunkShiftMs + 200) / 1000) * SAMPLE_RATE)
    }
}

interface StreamNemotronOptions extends DriverCallbacks {
    paths: NemotronModelPaths
    vadModelPath: string
    /** Reported on every segment. The shipped English export is monolingual. */
    language?: string
    /**
     * Passed to the recognizer as modelConfig.language for a MULTILINGUAL export (Nemotron 3.5
     * covers ~40 languages in one model). Undocumented in sherpa-onnx-node's JSDoc but read by the
     * addon; an unknown value falls back to "auto" with a warning from the native side. Ignored
     * entirely by the English-only export, which has no language embedding.
     */
    modelLanguage?: string
    /**
     * Merged into the OnlineRecognizer config, last. Exists so the benchmark harness can try
     * alternative model sets and decoding methods (hotwords need modified_beam_search plus a
     * bpe.model, neither of which the shipped Nemotron export supports) without forking this file.
     * Not set by the app.
     */
    recognizerOverrides?: Record<string, unknown>
    /**
     * Mel bins the export expects. Nemotron uses 128 and reads it from its own metadata, so the
     * value is advisory there; the Zipformer exports use 80 and do NOT carry it, so it is the
     * config that decides and a wrong one yields plausible-but-wrong words.
     */
    featureDim?: number
    /**
     * The export's encoder chunk shift in ms. Defaults to the shipped model's. Only set this when
     * pointing at a different export - every commit timing scales off it.
     */
    chunkShiftMs?: number
    /**
     * Minimum audio between decoder resets. reset() clears the hypothesis AND the RNN-T predictor
     * state - the decoder's memory of what it has just been saying - while the encoder cache
     * survives. Resetting at every utterance boundary was measurably costing transcription: the
     * same audio that decodes as "one Corinthians chapter three" on an unreset stream came out as
     * "one Corinthians cha three" with a reset shortly before it.
     *
     * Defaults to DEFAULT_RESET_INTERVAL_MS. 0 restores a reset at every utterance boundary.
     */
    resetIntervalMs?: number
    /** Injected by tests. Production loads the native addon lazily in start(). */
    sherpa?: any
}

export class NemotronStreamDriver implements TranscriptionDriver {
    private options: StreamNemotronOptions
    private timings: ReturnType<typeof gridTimings>

    private recognizer: any = null
    private vad: any = null
    /** ONE stream for the whole session. Recreating it is what the old path got wrong. */
    private stream: any = null

    private stopped = false
    private totalSamples = 0

    private inUtterance = false
    /** Absolute sample index at which the open utterance must be closed, 0 when none is pending. */
    private finalizeAtSample = 0

    // Emission is tracked in CHARACTERS of the hypothesis, not words. Greedy RNN-T emits BPE
    // pieces, so the trailing word grows in place ("Ephes" -> "Ephesians"); a word counter cannot
    // tell that apart from a new word and would silently drop the completion.
    private emittedChars = 0
    private lastText = ""
    private nextEmitStartMs = 0
    /** Absolute sample index at which the hypothesis last got longer - drives the static-tail rule. */
    private lastGrowthAtSample = 0
    /** Absolute sample index of the last decoder reset, for resetIntervalMs. */
    private lastResetAtSample = 0

    constructor(options: StreamNemotronOptions) {
        this.options = options
        this.timings = gridTimings(options.chunkShiftMs ?? NEMOTRON_CHUNK_SHIFT_MS)
    }

    async start(): Promise<void> {
        if (this.stopped) throw new Error("NemotronStreamDriver has already been stopped")

        // required lazily so the app still starts where the native addon fails to load
        // eslint-disable-next-line @typescript-eslint/no-var-requires
        const sherpa = this.options.sherpa || require("sherpa-onnx-node")
        const { paths, vadModelPath } = this.options

        const { recognizerOverrides } = this.options
        this.recognizer = new sherpa.OnlineRecognizer({
            // Nemotron declares feat_dim in its metadata and sherpa reads it from there, so this is
            // advisory for that export - but the Zipformer exports carry no feat_dim and the config
            // is what decides, so a wrong value there produces coherent, wrong words
            featConfig: { sampleRate: SAMPLE_RATE, featureDim: this.options.featureDim ?? 128 },
            modelConfig: {
                transducer: { encoder: paths.encoder, decoder: paths.decoder, joiner: paths.joiner },
                tokens: paths.tokens,
                numThreads: 2,
                provider: "cpu",
                debug: 0,
                ...(this.options.modelLanguage ? { language: this.options.modelLanguage } : {}),
                ...((recognizerOverrides?.modelConfig as Record<string, unknown>) || {})
            },
            // greedy is the only method this model supports, which also rules out hotword biasing
            // (both live behind modified_beam_search) - see the file header
            decodingMethod: "greedy_search",
            // boundaries come from Silero. Sherpa's own endpointing counts trailing BLANK FRAMES
            // from the decoder - it is not the energy gate driver.ts's header describes - but it
            // can only be evaluated at encoder-step boundaries, so its resolution is one 1120ms
            // chunk. Silero's 512-sample window is 32ms, and it also gates music and crowd noise.
            enableEndpoint: false,
            ...Object.fromEntries(Object.entries(recognizerOverrides || {}).filter(([key]) => key !== "modelConfig"))
        })

        this.stream = this.recognizer.createStream()

        this.vad = new sherpa.Vad(
            {
                sileroVad: {
                    model: vadModelPath,
                    threshold: VAD_THRESHOLD,
                    minSilenceDuration: VAD_MIN_SILENCE,
                    minSpeechDuration: VAD_MIN_SPEECH,
                    maxSpeechDuration: VAD_MAX_SPEECH,
                    windowSize: 512
                },
                sampleRate: SAMPLE_RATE,
                numThreads: 1,
                provider: "cpu",
                debug: 0
            },
            60
        )
    }

    async stop(): Promise<void> {
        if (this.stopped) return
        this.stopped = true

        // flush whatever was still being spoken so its text is not lost
        try {
            this.flushTail()
            if (this.inUtterance) this.closeUtterance()
        } catch (err) {
            console.error("[nemotron] Failed to flush the final utterance:", err)
        }

        this.recognizer = null
        this.vad = null
        this.stream = null
    }

    pushAudio(buffer: Uint8Array): void {
        if (this.stopped || !this.recognizer || !this.stream) return

        const samples = int16ToFloat32(buffer)
        if (!samples.length) return

        try {
            this.vad.acceptWaveform(samples)

            // EVERY push reaches the recognizer, silence included. This is the invariant the whole
            // file rests on: the encoder cache is only continuous if the audio is continuous, and
            // skipping non-speech would re-introduce the priming cost at each utterance start.
            this.stream.acceptWaveform({ sampleRate: SAMPLE_RATE, samples })
            while (this.recognizer.isReady(this.stream)) this.recognizer.decode(this.stream)

            this.totalSamples += samples.length

            if (this.vad.isDetected()) {
                if (!this.inUtterance) {
                    this.inUtterance = true
                    this.nextEmitStartMs = Math.max(this.nextEmitStartMs, this.currentMs())
                }
                // speech resumed inside the defer window - it was a pause, not an end
                this.finalizeAtSample = 0
            }

            // drain the VAD's own queue; a close arms the deferred boundary
            let closed = false
            while (!this.vad.isEmpty()) {
                this.vad.pop()
                closed = true
            }
            if (closed && this.inUtterance && !this.finalizeAtSample) {
                this.finalizeAtSample = this.totalSamples + this.timings.closeDeferSamples
            }

            if (this.finalizeAtSample && this.totalSamples >= this.finalizeAtSample) {
                this.finalizeAtSample = 0
                this.closeUtterance()
            } else if (this.inUtterance) {
                this.emitFromHypothesis(false)
            }
        } catch (err) {
            this.options.onError(String((err as Error)?.message || err))
        }
    }

    // EMISSION

    private currentMs(): number {
        return Math.round((this.totalSamples / SAMPLE_RATE) * 1000)
    }

    private readText(): string {
        return ((this.recognizer.getResult(this.stream).text || "") as string).trim()
    }

    /**
     * Push silence so the audio after the last encoder step still gets decoded. Only needed at
     * stop(): mid-session an utterance always closes a full chunk shift of real audio past the
     * speech end, but at stop() the audio simply ends and the remainder would never be decoded.
     */
    private flushTail() {
        if (!this.recognizer || !this.stream) return

        this.stream.acceptWaveform({ sampleRate: SAMPLE_RATE, samples: new Float32Array(this.timings.closeDeferSamples) })
        while (this.recognizer.isReady(this.stream)) this.recognizer.decode(this.stream)
    }

    /**
     * Emit whatever the hypothesis has gained since the last call.
     *
     * While the utterance is open the trailing word is held back, because a greedy RNN-T emits
     * token by token and the last word may be a BPE fragment mid-emission ("Ephes"). It goes to
     * the interim display instead and is committed by the next call, one encoder step later.
     * Note that only the LAST word pays that - in driver.ts every word paid a flat 1.2s.
     */
    private emitFromHypothesis(final: boolean) {
        const text = this.readText()
        if (!text) {
            if (!final) this.options.onInterim?.("")
            return
        }

        this.assertPrefix(text)
        if (text.length > this.lastText.length) this.lastGrowthAtSample = this.totalSamples
        this.lastText = text

        // A greedy RNN-T can lock into a cycle and emit the same phrase indefinitely, because the
        // predictor's own output is its next input - seen filling a live transcript with "and he
        // saith the LORD" over and over. Nothing in the audio pulls it out; only clearing the
        // decoder state does, so this both keeps the first occurrence and breaks the cycle.
        const loopAt = findRepeatedTail(text)
        if (loopAt >= 0) {
            const keep = text.slice(0, loopAt).trimEnd()
            if (keep.length > this.emittedChars) {
                const candidate = keep.slice(this.emittedChars).trim()
                this.emittedChars = keep.length
                if (candidate) this.emitText(candidate, false)
            }
            console.warn(`[nemotron] decoder was repeating ${JSON.stringify(text.slice(loopAt).slice(0, 60))} - clearing its state to break the cycle`)
            this.resetDecoder()
            this.options.onInterim?.("")
            return
        }

        // the trailing word is held back unless the utterance is closing, or the decoder has gone
        // a full encoder step without adding anything - at which point it is as settled as it will
        // ever be and holding it costs latency for nothing
        const settled = final || this.totalSamples - this.lastGrowthAtSample >= this.timings.staticTailSamples
        const lastBoundary = text.lastIndexOf(" ")
        const commitTo = settled ? text.length : lastBoundary

        if (commitTo > this.emittedChars) {
            const candidate = text.slice(this.emittedChars, commitTo).trim()
            this.emittedChars = commitTo
            if (candidate) this.emitText(candidate, final)
            else if (final && this.emittedChars > 0) this.emitBoundary()
        } else if (final && this.emittedChars > 0) {
            // the utterance ended without new words, but the display still has to close its line
            this.emitBoundary()
        }

        this.options.onInterim?.(final ? "" : text.slice(this.emittedChars).trim())
    }

    /** Clear the decoder's state and the emission bookkeeping that tracks its text. */
    private resetDecoder() {
        this.recognizer.reset(this.stream)
        this.lastResetAtSample = this.totalSamples
        this.emittedChars = 0
        this.lastText = ""
    }

    /** Close the open utterance: commit the remainder, then clear the hypothesis. */
    private closeUtterance() {
        this.emitFromHypothesis(true)

        this.inUtterance = false
        this.lastGrowthAtSample = this.totalSamples

        // clears the decoded text but NOT the encoder cache (verified - see the file header), so
        // the next utterance starts warm and its first word does not wait out the priming window.
        // It DOES clear the RNN-T predictor state, which costs transcription accuracy, so an
        // interval keeps that context across utterance boundaries - see resetIntervalMs.
        const interval = this.options.resetIntervalMs ?? DEFAULT_RESET_INTERVAL_MS
        if (this.totalSamples - this.lastResetAtSample >= (interval / 1000) * SAMPLE_RATE) this.resetDecoder()
    }

    /**
     * The monotonicity guard. Greedy RNN-T cannot retract, and 120s of continuous preaching never
     * produced a non-prefix result - but sherpa's homophone replacer and rule_fsts post-processors
     * would, and neither is enabled here today. If that ever changes this is where it surfaces,
     * loudly, instead of silently duplicating text in the transcript.
     */
    private assertPrefix(text: string) {
        if (!this.lastText || text.startsWith(this.lastText)) return

        console.warn(`[nemotron] hypothesis was revised, not extended - emission restarts from the new text. was ${JSON.stringify(this.lastText.slice(-40))}, now ${JSON.stringify(text.slice(-40))}`)
        // treat the revision as a fresh hypothesis: already-emitted words are never retracted, so
        // the only safe move is to re-anchor and let the next extension carry on from here
        this.emittedChars = text.length
    }

    private emitText(text: string, utteranceEnd: boolean) {
        if (!text) {
            if (utteranceEnd) this.emitBoundary()
            return
        }

        const endMs = this.currentMs()
        const segment: TranscriberSegment = { text, startMs: this.nextEmitStartMs, endMs }
        if (utteranceEnd) segment.utteranceEnd = true
        this.nextEmitStartMs = endMs

        if (this.options.language) segment.language = this.options.language
        this.options.onSegment(segment)
    }

    /** An utterance that ends with no new words still ends - the display closes its line on this. */
    private emitBoundary() {
        const endMs = this.currentMs()
        const segment: TranscriberSegment = { text: "", startMs: this.nextEmitStartMs, endMs, utteranceEnd: true }
        this.nextEmitStartMs = endMs
        if (this.options.language) segment.language = this.options.language
        this.options.onSegment(segment)
    }
}

/** Int16 LE PCM bytes (as sent over IPC) to the Float32 samples sherpa expects. */
function int16ToFloat32(buffer: Uint8Array): Float32Array {
    const view = new DataView(buffer.buffer, buffer.byteOffset, buffer.byteLength)
    const count = Math.floor(buffer.byteLength / 2)
    const samples = new Float32Array(count)
    for (let i = 0; i < count; i++) samples[i] = view.getInt16(i * 2, true) / 32768
    return samples
}
