// AI BENCH - derive scores from a RunResult's event log.
//
// Nothing here touches audio or an engine, so a new metric is a re-analysis rather than a re-run.
//
// The headline metric is `commitLag`, and it deserves an explanation because it is not a standard
// ASR number. FreeShow shows two streams: greyed INTERIM text (display only) and FINAL segments
// (the only thing scripture detection ever sees - SpeechToTextManager.onInterim's comment is
// explicit that detection never sees interim). So a word being *visible* and a word being
// *actionable* are different events, and the gap between them is time the feature spends knowing
// the answer without being allowed to use it. That gap is a policy choice, not a model
// limitation, which is exactly why it is worth measuring separately from everything else.

import { align, normalizeForWer, vocabularyErrorRate, type Alignment } from "./align"
import type { EmissionEvent, RunResult } from "./runner"
import { describe as describeDistribution, type Distribution } from "./stats"

export interface CommitLagMetrics {
    /** Per final word: ms of audio between it first being visible as interim and being emitted. */
    lag: Distribution
    /** Words that were emitted as final without ever appearing as interim first. */
    neverInterim: number
    /** Words counted. */
    words: number
}

export interface WerMetrics {
    wer: number
    substitutions: number
    deletions: number
    insertions: number
    refLength: number
    alignment: Alignment
}

export interface RunMetrics {
    fixtureId: string
    variantId: string
    audioDurationMs: number
    commitLag?: CommitLagMetrics
    wer?: WerMetrics
    vocabulary?: { errors: number; total: number; rate: number; missed: string[] }
    /** Interim text that reappeared after the same words were already finalized - visible flicker. */
    interimEchoes: { audioMs: number; text: string }[]
    /**
     * CPU seconds consumed per second of audio. Above 1.0 the engine cannot keep up on this
     * machine. Derived from process.cpuUsage(), not wall time - see PacerStats.cpuMs for why that
     * distinction is the difference between a real number and a measure of how busy the desktop was.
     */
    decodeCostRatio: number
    /** The same ratio computed from wall time. Only meaningful on an otherwise idle machine. */
    wallCostRatio: number
    startupMs: number
    errors: string[]
}

/** Running view of what the user can see: finalized words plus the current interim tail. */
function visibleTokensAt(events: EmissionEvent[], upTo: number): string[] {
    const finals: string[] = []
    let interim: string[] = []

    for (let i = 0; i <= upTo; i++) {
        const event = events[i]
        if (event.kind === "segment") {
            if (event.text) finals.push(...normalizeForWer(event.text))
            interim = []
        } else {
            interim = normalizeForWer(event.text)
        }
    }
    return [...finals, ...interim]
}

/**
 * For every finalized word, the audio time at which it first became visible (as interim or as the
 * final itself) versus the audio time at which it was committed.
 *
 * Matching is positional: token k of the visible stream against token k of the final stream. A
 * revision earlier in the text shifts later positions, and treating that as "not yet seen" is the
 * honest reading - the word at that position genuinely was not settled.
 */
export function commitLag(result: RunResult): CommitLagMetrics {
    const finalWordCommitMs: number[] = []
    const finalWords: string[] = []

    for (const event of result.events) {
        if (event.kind !== "segment" || !event.text) continue
        for (const token of normalizeForWer(event.text)) {
            finalWords.push(token)
            finalWordCommitMs.push(event.audioMs)
        }
    }

    const firstVisibleMs = new Array<number>(finalWords.length).fill(Number.NaN)
    for (let i = 0; i < result.events.length; i++) {
        const visible = visibleTokensAt(result.events, i)
        const at = result.events[i].audioMs
        for (let k = 0; k < Math.min(visible.length, finalWords.length); k++) {
            if (Number.isNaN(firstVisibleMs[k]) && visible[k] === finalWords[k]) firstVisibleMs[k] = at
        }
    }

    const lags: number[] = []
    let neverInterim = 0
    for (let k = 0; k < finalWords.length; k++) {
        if (Number.isNaN(firstVisibleMs[k])) {
            neverInterim++
            continue
        }
        const lag = finalWordCommitMs[k] - firstVisibleMs[k]
        if (lag > 0) lags.push(lag)
        else if (lag === 0) {
            lags.push(0)
            neverInterim++ // committed in the same event it first appeared - no interim lead at all
        }
    }

    return { lag: describeDistribution(lags), neverInterim, words: finalWords.length }
}

/**
 * Interim text that repeats words already finalized. The seam stitch (speech/seam.ts) keeps these
 * out of the committed transcript, but the user still sees the word twice on screen, so it is a
 * real defect that WER cannot see.
 */
export function interimEchoes(result: RunResult): { audioMs: number; text: string }[] {
    const echoes: { audioMs: number; text: string }[] = []
    const finalized: string[] = []

    for (const event of result.events) {
        if (event.kind === "segment") {
            if (event.text) finalized.push(...normalizeForWer(event.text))
            continue
        }

        const tokens = normalizeForWer(event.text)
        if (!tokens.length || finalized.length < tokens.length) continue

        // an interim that starts by repeating the tail of what was already committed
        const tail = finalized.slice(finalized.length - tokens.length)
        const overlap = tokens.every((token, index) => token === tail[index])
        if (overlap) echoes.push({ audioMs: event.audioMs, text: event.text })
    }

    return echoes
}

export function scoreRun(result: RunResult, reference?: string, vocabulary?: Set<string>): RunMetrics {
    const metrics: RunMetrics = {
        fixtureId: result.fixtureId,
        variantId: result.variantId,
        audioDurationMs: result.audioDurationMs,
        commitLag: commitLag(result),
        interimEchoes: interimEchoes(result),
        decodeCostRatio: result.audioDurationMs ? result.pacer.cpuMs / result.audioDurationMs : 0,
        wallCostRatio: result.audioDurationMs ? result.pacer.pushBlockedMs / result.audioDurationMs : 0,
        startupMs: result.startupMs,
        errors: result.errors
    }

    if (reference) {
        const alignment = align(normalizeForWer(reference), normalizeForWer(result.hypothesis))
        metrics.wer = {
            wer: alignment.wer,
            substitutions: alignment.substitutions,
            deletions: alignment.deletions,
            insertions: alignment.insertions,
            refLength: alignment.refLength,
            alignment
        }
        if (vocabulary?.size) metrics.vocabulary = vocabularyErrorRate(alignment, vocabulary)
    }

    return metrics
}
