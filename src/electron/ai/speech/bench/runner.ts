// AI BENCH - drive one fixture through one engine variant and record every emission.
//
// This is the skeleton the metrics layers build on. It deliberately computes NO scores: it
// produces the raw event log (what was emitted, at which audio timestamp, at which wall
// timestamp) and the resource numbers. WER, detection recall and latency percentiles are all
// derived from this log afterwards, so adding a metric never means re-running a decode.
//
// The one judgement it does make is which clock a number belongs to. See pacer.ts: audio-time is
// a property of the engine, wall-time is a property of the machine. Both are recorded on every
// event; the report layer refuses to print wall-time from a "max" run.

import { AudioClock, pace, type PacerStats, type PaceMode } from "./pacer"
import { createDriver, type EngineVariant } from "./engines"
import type { TranscriberSegment } from "../types"
import { readWav16kMono } from "./wav"

export interface EmissionEvent {
    kind: "segment" | "interim"
    text: string
    /** How much audio had been pushed when this arrived. Machine-independent. */
    audioMs: number
    /** Wall clock since the run started. Only meaningful when mode is "rt". */
    wallMs: number
    /** Segments only - the driver's own idea of when this text was spoken. */
    startMs?: number
    endMs?: number
    utteranceEnd?: boolean
    music?: boolean
}

export interface RunResult {
    fixtureId: string
    fixturePath: string
    variantId: string
    mode: PaceMode
    /** Set when the fixture was not natively 16 kHz - flagged because it taints WER comparisons. */
    resampledFrom?: number
    audioDurationMs: number
    /** Every emission, in arrival order. The substrate for every metric. */
    events: EmissionEvent[]
    /** Concatenated final segment text, in arrival order - the hypothesis transcript. */
    hypothesis: string
    pacer: PacerStats
    /** Wall time spent in start() - model load. */
    startupMs: number
    errors: string[]
    /** Stamped so the report layer can refuse cross-machine comparisons silently. */
    platform: string
    arch: string
}

export interface RunOptions {
    fixtureId: string
    fixturePath: string
    variant: EngineVariant
    mode: PaceMode
    /** Fraction of audio chunks to drop - see PacerOptions.dropRate. */
    dropRate?: number
    language?: string
    onProgress?: (audioPushedMs: number, totalMs: number) => void
}

export async function runFixture(options: RunOptions): Promise<RunResult> {
    const wav = readWav16kMono(options.fixturePath)
    const clock = new AudioClock()

    const events: EmissionEvent[] = []
    const errors: string[] = []
    let runStartedAt = Date.now()

    const record = (event: Omit<EmissionEvent, "audioMs" | "wallMs">) => {
        events.push({ ...event, audioMs: clock.audioPushedMs, wallMs: Date.now() - runStartedAt })
    }

    const driver = createDriver(
        options.variant,
        {
            onSegment: (segment: TranscriberSegment) =>
                record({
                    kind: "segment",
                    text: segment.text,
                    startMs: segment.startMs,
                    endMs: segment.endMs,
                    utteranceEnd: segment.utteranceEnd,
                    music: segment.music
                }),
            onInterim: (text: string) => record({ kind: "interim", text }),
            // an engine error does not abort the run: a partial result plus the message is more
            // informative than a thrown exception, and a crash mid-fixture is itself a finding
            onError: (message: string) => errors.push(message)
        },
        options.language
    )

    const startupStartedAt = Date.now()
    await driver.start()
    const startupMs = Date.now() - startupStartedAt

    // the run clock starts AFTER model load, so a 1.1s load does not inflate every event's wallMs
    runStartedAt = Date.now()

    let pacer: PacerStats
    try {
        pacer = await pace(driver, wav.samples, clock, { mode: options.mode, dropRate: options.dropRate, onProgress: options.onProgress })
    } finally {
        // stop() flushes the utterance still open, and those segments must land in the log
        await driver.stop()
    }

    return {
        fixtureId: options.fixtureId,
        fixturePath: options.fixturePath,
        variantId: options.variant.id,
        mode: options.mode,
        ...(wav.resampledFrom ? { resampledFrom: wav.resampledFrom } : {}),
        audioDurationMs: wav.durationMs,
        events,
        hypothesis: events
            .filter((event) => event.kind === "segment" && event.text)
            .map((event) => event.text)
            .join(" ")
            .replace(/\s+/g, " ")
            .trim(),
        pacer,
        startupMs,
        errors,
        platform: process.platform,
        arch: process.arch
    }
}
