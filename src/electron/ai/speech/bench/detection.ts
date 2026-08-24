// AI BENCH - the product metric.
//
// WER says how many words the engine got right. This says whether the feature worked: the speaker
// said "Ephesians chapter two verse eight" and the verse either reached the screen or it did not,
// and the delay between the phrase ending and the reference becoming actionable is what an
// operator experiences as the feature being fast or useless.
//
// It is a replay, not a second decode: the real DetectionCoordinator is driven from a RunResult's
// event log, so no audio, no engine, no electron and no IPC are involved and detection can be
// re-scored or re-tuned against a decode that already happened. The segments handed over are the
// ones ai/index.ts lets through (music and textless segments are dropped there, and interim text
// never reaches detection at all), so what is scored is the whole live path - engine emission
// timing plus tier 1 matching plus the emission cooldown - rather than the matcher in isolation.

import type { AiScriptureBook, AiScriptureState, DetectedReference } from "../../../../types/ai/AiScripture"
import { DetectionCoordinator, type AiScriptureAnchor } from "../../scripture/detection/coordinator"
import type { ExpectedReference } from "./fixtures"
import type { RunResult } from "./runner"
import { describe as describeDistribution, type Distribution } from "./stats"

// The renderer hands the coordinator the book names of the bibles the user selected. A fixture has
// no bible, so the bench stands in with the plain English canon - deliberately plain, because a
// recall number that only holds up with a hand-tuned name table is measuring the table. Canon
// numbering is also what ExpectedReference.book means. "/" separates spoken alternates; misheard
// forms are added by the coordinator itself (detection/asrRepairs.ts).
const CANON_BOOK_NAMES = [
    "Genesis",
    "Exodus",
    "Leviticus",
    "Numbers",
    "Deuteronomy",
    "Joshua",
    "Judges",
    "Ruth",
    "1 Samuel",
    "2 Samuel",
    "1 Kings",
    "2 Kings",
    "1 Chronicles",
    "2 Chronicles",
    "Ezra",
    "Nehemiah",
    "Esther",
    "Job",
    "Psalms/Psalm",
    "Proverbs",
    "Ecclesiastes",
    "Song of Solomon/Song of Songs",
    "Isaiah",
    "Jeremiah",
    "Lamentations",
    "Ezekiel",
    "Daniel",
    "Hosea",
    "Joel",
    "Amos",
    "Obadiah",
    "Jonah",
    "Micah",
    "Nahum",
    "Habakkuk",
    "Zephaniah",
    "Haggai",
    "Zechariah",
    "Malachi",
    "Matthew",
    "Mark",
    "Luke",
    "John",
    "Acts",
    "Romans",
    "1 Corinthians",
    "2 Corinthians",
    "Galatians",
    "Ephesians",
    "Philippians",
    "Colossians",
    "1 Thessalonians",
    "2 Thessalonians",
    "1 Timothy",
    "2 Timothy",
    "Titus",
    "Philemon",
    "Hebrews",
    "James",
    "1 Peter",
    "2 Peter",
    "1 John",
    "2 John",
    "3 John",
    "Jude",
    "Revelation"
]

export const BENCH_BOOKS: AiScriptureBook[] = CANON_BOOK_NAMES.map((names, index) => ({ number: index + 1, canonNumber: index + 1, names: names.split("/") }))

export interface DetectionReplayOptions {
    /** Defaults to the plain English canon. Pass a real bible's table to measure that instead. */
    books?: AiScriptureBook[]
    /**
     * Tier 2. OFF by default, and that is the point: an LLM's detections and the local matcher's
     * land in the same callback, so a mixed number cannot tell a regex regression from a better
     * prompt. Measure one tier at a time, then measure both.
     */
    llm?: { provider: string; model: string } | null
    getApiKey?: (providerId: string) => string
    cooldownSeconds?: number
    /** The passage live on the output, when the fixture assumes one - bare "verse N" resolves against it. */
    anchor?: AiScriptureAnchor
    /**
     * Awaited after every segment. Only tier 2 needs it: a detection from an LLM lands a turn (or
     * a network round trip) later, and the replay has to give it that turn. Defaults to a
     * macrotask flush when `llm` is set, which covers a mocked provider but not a real one.
     */
    settle?: () => Promise<void>
}

export interface DetectionEmission {
    /** Audio time the reference became actionable: when the segment carrying it arrived. */
    audioMs: number
    reference: DetectedReference
}

export interface DetectionReplay {
    fixtureId: string
    variantId: string
    audioDurationMs: number
    /** Segments that actually reached detection - music & textless ones are dropped, as in production. */
    segmentsFed: number
    detections: DetectionEmission[]
    statuses: { state: AiScriptureState; message?: string }[]
}

const flushMacrotask = () => new Promise<void>((resolve) => setTimeout(resolve, 0))

// The audio clock below is installed on the global Date.now, so replays cannot overlap: two in
// flight would read each other's audio time, and whichever unwound last would restore the OTHER
// one's patch and leave the process on a frozen clock for good. A bench replays fixtures one at a
// time; a Promise.all over them fails here instead of silently.
let replayInProgress = false

export async function replayDetection(result: RunResult, options: DetectionReplayOptions = {}): Promise<DetectionReplay> {
    if (replayInProgress) throw new Error("replayDetection cannot run concurrently with another replay - it patches Date.now")

    const detections: DetectionEmission[] = []
    const statuses: DetectionReplay["statuses"] = []
    let audioMs = 0

    const coordinator = new DetectionCoordinator({
        books: options.books ?? BENCH_BOOKS,
        llm: options.llm ?? null,
        getApiKey: options.getApiKey ?? (() => ""),
        cooldownSeconds: options.cooldownSeconds,
        onDetection: (reference) => detections.push({ audioMs, reference }),
        onStatus: (state, extra) => statuses.push({ state, ...(extra?.message ? { message: extra.message } : {}) })
    })
    if (options.anchor) coordinator.updateContext(options.anchor)

    const settle = options.settle ?? (options.llm ? flushMacrotask : null)

    // The cooldown that suppresses a re-emitted reference is measured on Date.now(), and a replay
    // of ten minutes of audio finishes in milliseconds - so a passage the speaker returns to after
    // five minutes would look suppressed here while a live service re-projects it, understating
    // recall on exactly the fixtures that matter most. Running the replay with Date.now() driven
    // by the audio clock is what makes the cooldown behave the way the operator sees it.
    // Restored in the finally; with tier 2 on the patch also spans the settle await, so anything
    // else running in this process during a replay sees the shifted clock.
    const realNow = Date.now
    const base = realNow()
    Date.now = () => base + audioMs
    replayInProgress = true

    let segmentsFed = 0
    try {
        for (const event of result.events) {
            if (event.kind !== "segment") continue // detection never sees interim text
            if (event.music || !event.text) continue // sung content & utterance-boundary markers are dropped upstream

            audioMs = event.audioMs
            segmentsFed++

            // the engine's own timings drive the rolling transcript window, as they do in production;
            // audioMs stands in for a driver that does not report them
            const endMs = event.endMs ?? event.audioMs
            // utteranceEnd is load bearing, not decoration: tier 1 holds a reference sitting at the
            // very end of the transcript because it may still be being spoken, and this marker is
            // what tells it the speaker stopped. Dropping it made every held reference wait for the
            // next words instead, which inflated measured detection latency by seconds.
            coordinator.onTranscriptSegment({ text: event.text, startMs: event.startMs ?? endMs, endMs, utteranceEnd: event.utteranceEnd })

            if (settle) await settle()
        }
    } finally {
        replayInProgress = false
        Date.now = realNow
        coordinator.stop()
    }

    return { fixtureId: result.fixtureId, variantId: result.variantId, audioDurationMs: result.audioDurationMs, segmentsFed, detections, statuses }
}

// SCORING

export interface DetectionMatch {
    expected: ExpectedReference
    detected: DetectedReference
    audioMs: number
    /** audioMs - expected.phraseEndMs. The headline number: how late the verse could be shown. */
    latencyMs: number
}

export interface DetectionScore {
    fixtureId: string
    variantId: string
    audioDurationMs: number
    expectedCount: number
    detectionCount: number
    matched: number
    /** The passage was already credited to an earlier spoken phrase. Neither earned nor punished. */
    duplicates: number
    falsePositives: number
    /** null - not 0 - when the fixture has no expected references: reference-free audio has no recall. */
    recall: number | null
    /** null when nothing was detected at all. */
    precision: number | null
    falsePositivesPerMinute: number
    latencyMs: Distribution
    /** Matched, but the verse range ended somewhere else ("verses 28 to 30" projected as verse 28). */
    rangeMismatches: number
    matches: DetectionMatch[]
    missed: ExpectedReference[]
    spurious: DetectionEmission[]
}

export interface DetectionScoreOptions {
    /**
     * How far before its phrase ends a detection may still be credited to it. phraseEndMs is
     * hand-marked for anything but a generated fixture, and a detection cannot precede the words
     * that caused it - so this is slack for the person who marked the audio, not for the engine.
     */
    earlyToleranceMs?: number
    /**
     * How long after its phrase ends a detection may still be credited to it. Without a bound the
     * earliest unconsumed match wins at any distance, so an unmarked re-mention forty minutes later
     * scores as a hit and recall - the headline number - is overstated. A verse the preacher has
     * long since moved past is not a success at any latency.
     */
    lateToleranceMs?: number
}

const DEFAULT_EARLY_TOLERANCE_MS = 1500
/**
 * A reference is quoted, then read out and expounded; a detection landing inside that window is
 * still useful on screen, one landing after it is not. Deliberately generous - the point is to
 * exclude the unrelated re-mention, not to double as a latency threshold, which the latency
 * distribution reports honestly on its own.
 */
const DEFAULT_LATE_TOLERANCE_MS = 30000

/**
 * book + chapter + verseStart, nothing else.
 *
 * verseEnd is deliberately not compared. The manifest records the passage the operator wanted on
 * screen, and whether the range end survived ("verses 28 to 30" arriving as verse 28) changes which
 * verses are highlighted, not whether the feature found the passage - so it is reported separately
 * as rangeMismatches instead of silently costing recall. A wrong verseStart is a different passage
 * and never matches.
 */
function sameReference(detected: DetectedReference, expected: ExpectedReference): boolean {
    return detected.bookNumber === expected.book && detected.chapter === expected.chapter && detected.verseStart === expected.verseStart
}

export function scoreDetection(replay: DetectionReplay, expected: ExpectedReference[], options: DetectionScoreOptions = {}): DetectionScore {
    const tolerance = options.earlyToleranceMs ?? DEFAULT_EARLY_TOLERANCE_MS
    const lateTolerance = options.lateToleranceMs ?? DEFAULT_LATE_TOLERANCE_MS
    const references = [...expected].sort((a, b) => a.phraseEndMs - b.phraseEndMs)
    const detections = [...replay.detections].sort((a, b) => a.audioMs - b.audioMs)

    // Assignment, in spoken order, of the earliest detection that could belong to each phrase. One
    // detection is credited to at most one expected reference, so a passage the manifest lists
    // twice needs two detections to score full recall - which is what a live service would produce
    // once the cooldown has expired between the two mentions.
    const consumed = new Set<number>()
    const matches: DetectionMatch[] = []
    const missed: ExpectedReference[] = []

    for (const reference of references) {
        const earliest = reference.phraseEndMs - tolerance
        const latest = reference.phraseEndMs + lateTolerance
        const index = detections.findIndex((detection, at) => !consumed.has(at) && detection.audioMs >= earliest && detection.audioMs <= latest && sameReference(detection.reference, reference))
        if (index < 0) {
            missed.push(reference)
            continue
        }

        consumed.add(index)
        matches.push({ expected: reference, detected: detections[index].reference, audioMs: detections[index].audioMs, latencyMs: detections[index].audioMs - reference.phraseEndMs })
    }

    // Leftovers. A repeat of a passage the manifest already accounted for is not a false positive:
    // the manifest lists the phrases someone marked, not every time the preacher said it again, so
    // penalizing the repeat would make precision depend on how thorough the marking was. Everything
    // else - including a detection that fired before the phrase it names was spoken - is spurious.
    const spurious: DetectionEmission[] = []
    let duplicates = 0
    detections.forEach((detection, at) => {
        if (consumed.has(at)) return
        // A late detection of a passage the manifest knows about is still a repeat, not a false
        // positive - the preacher came back to it, and the manifest only lists marked phrases. The
        // window above governs what counts as ANSWERING a phrase; this only governs blame.
        const repeat = references.some((reference) => detection.audioMs >= reference.phraseEndMs - tolerance && sameReference(detection.reference, reference))
        if (repeat) duplicates++
        else spurious.push(detection)
    })

    const matched = matches.length
    const falsePositives = spurious.length
    const judged = matched + falsePositives
    const minutes = replay.audioDurationMs / 60000
    const rangeMismatches = matches.filter((match) => match.detected.verseEnd !== (match.expected.verseEnd ?? match.expected.verseStart)).length

    return {
        fixtureId: replay.fixtureId,
        variantId: replay.variantId,
        audioDurationMs: replay.audioDurationMs,
        expectedCount: references.length,
        detectionCount: detections.length,
        matched,
        duplicates,
        falsePositives,
        recall: references.length ? matched / references.length : null,
        precision: judged ? matched / judged : null,
        falsePositivesPerMinute: minutes > 0 ? falsePositives / minutes : 0,
        latencyMs: describeDistribution(matches.map((match) => match.latencyMs)),
        rangeMismatches,
        matches,
        missed,
        spurious
    }
}

export interface DetectionSummary {
    fixtures: number
    /** Fixtures carrying at least one expected reference - the only ones recall is computed over. */
    referenceFixtures: number
    audioDurationMs: number
    expectedCount: number
    matched: number
    falsePositives: number
    recall: number | null
    precision: number | null
    falsePositivesPerMinute: number
    latencyMs: Distribution
}

/**
 * Pooled over a fixture set. Recall pools references (a fixture with six references weighs six
 * times one with a single reference - a per-fixture average would let a one-reference clip decide
 * the headline), while false positives per minute pools over EVERY fixture's audio, including the
 * reference-free ones. Those clips are the only honest measure of how often the feature interrupts
 * a service that never asked for a verse.
 */
export function summarizeDetection(scores: DetectionScore[]): DetectionSummary {
    const expectedCount = scores.reduce((total, score) => total + score.expectedCount, 0)
    const matched = scores.reduce((total, score) => total + score.matched, 0)
    const falsePositives = scores.reduce((total, score) => total + score.falsePositives, 0)
    const audioDurationMs = scores.reduce((total, score) => total + score.audioDurationMs, 0)
    const judged = matched + falsePositives
    const minutes = audioDurationMs / 60000

    return {
        fixtures: scores.length,
        referenceFixtures: scores.filter((score) => score.expectedCount > 0).length,
        audioDurationMs,
        expectedCount,
        matched,
        falsePositives,
        recall: expectedCount ? matched / expectedCount : null,
        precision: judged ? matched / judged : null,
        falsePositivesPerMinute: minutes > 0 ? falsePositives / minutes : 0,
        latencyMs: describeDistribution(scores.flatMap((score) => score.matches.map((match) => match.latencyMs)))
    }
}
