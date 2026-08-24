// AI BENCH - detection metric tests. Synthetic event logs only: no model, no audio, no network.

import { beforeEach, describe, expect, it, vi } from "vitest"

// the coordinator's tier 2 composes the real AI API providers - no test here may reach them
const { mockDetectScripture } = vi.hoisted(() => ({ mockDetectScripture: vi.fn() }))
vi.mock("../../scripture/llmTalkScripture", () => ({
    getLLMScriptureProvider: () => ({ detectScripture: mockDetectScripture })
}))

import type { DetectedReference } from "../../../../types/ai/AiScripture"
import { replayDetection, scoreDetection, summarizeDetection, type DetectionEmission, type DetectionReplay } from "./detection"
import type { ExpectedReference } from "./fixtures"
import type { EmissionEvent, RunResult } from "./runner"

type LogEntry = Partial<EmissionEvent> & { audioMs: number }

function runResult(entries: LogEntry[], audioDurationMs = 60000): RunResult {
    return {
        fixtureId: "fx",
        fixturePath: "fx.wav",
        variantId: "nemotron-stream",
        mode: "max",
        audioDurationMs,
        events: entries.map((entry) => ({ kind: "segment", text: "", wallMs: 0, endMs: entry.audioMs, ...entry, utteranceEnd: true })),
        hypothesis: "",
        pacer: { audioPushedMs: audioDurationMs, wallMs: 0, pushBlockedMs: 0, pushBlockedP50: 0, pushBlockedP99: 0, pushBlockedMax: 0, driftMs: 0 },
        startupMs: 0,
        errors: [],
        platform: "test",
        arch: "test"
    }
}

function expected(over: Partial<ExpectedReference> = {}): ExpectedReference {
    return { book: 43, chapter: 3, verseStart: 16, phrase: "john chapter three verse sixteen", phraseEndMs: 3000, ...over }
}

function emission(audioMs: number, reference: Partial<DetectedReference> & { bookNumber: number; chapter: number; verseStart: number }): DetectionEmission {
    return {
        audioMs,
        reference: { id: "d" + audioMs, book: "", verseEnd: reference.verseStart, confidence: "high", type: "explicit", source: "regex", timestamp: 0, ...reference }
    }
}

function replayOf(detections: DetectionEmission[], audioDurationMs = 60000): DetectionReplay {
    return { fixtureId: "fx", variantId: "v", audioDurationMs, segmentsFed: detections.length, detections, statuses: [] }
}

describe("bench/detection replay", () => {
    beforeEach(() => mockDetectScripture.mockReset())

    it("feeds only the segments production feeds, and stamps the detection with that segment's audio time", async () => {
        const replay = await replayDetection(
            runResult([
                { kind: "interim", text: "turn to john chapter three verse sixteen", audioMs: 2600 },
                { kind: "segment", text: "romans chapter eight verse twenty eight", audioMs: 3000, music: true, utteranceEnd: true },
                { kind: "segment", text: "", audioMs: 3400, utteranceEnd: true },
                { kind: "segment", text: "please turn to john chapter three verse sixteen", audioMs: 4200, utteranceEnd: true }
            ])
        )

        expect(replay.segmentsFed).toBe(1)
        expect(replay.detections).toHaveLength(1)
        expect(replay.detections[0].audioMs).toBe(4200)
        expect(replay.detections[0].reference).toMatchObject({ bookNumber: 43, chapter: 3, verseStart: 16, source: "regex" })
    })

    it("re-emits a passage the speaker returns to, because the cooldown runs on the audio clock", async () => {
        // the replay itself takes milliseconds - on the wall clock the 90 s cooldown would swallow
        // the second mention and the fixture would score 50% recall for something that works live
        const replay = await replayDetection(
            runResult(
                [
                    { text: "open your bibles to john chapter three verse sixteen", audioMs: 5000 },
                    { text: "back to john chapter three verse sixteen", audioMs: 200000 }
                ],
                300000
            )
        )

        expect(replay.detections.map((detection) => detection.audioMs)).toEqual([5000, 200000])
    })

    it("keeps the cooldown suppressing a repeat inside the window", async () => {
        const replay = await replayDetection(
            runResult([
                { text: "open your bibles to john chapter three verse sixteen", audioMs: 5000 },
                { text: "again john chapter three verse sixteen", audioMs: 25000 }
            ])
        )

        expect(replay.detections).toHaveLength(1)
    })

    it("leaves tier 2 off unless it is asked for", async () => {
        const replay = await replayDetection(runResult([{ text: "for by grace you have been saved through faith and that not of yourselves it is the gift of god not of works", audioMs: 8000 }]))

        expect(mockDetectScripture).not.toHaveBeenCalled()
        expect(replay.detections).toHaveLength(0)
    })

    it("records a tier 2 detection when the LLM is enabled", async () => {
        mockDetectScripture.mockResolvedValue({ references: [{ book: "Ephesians", bookNumber: 49, chapter: 2, verseStart: 8, verseEnd: 8, confidence: "high", type: "quoted" }] })

        const replay = await replayDetection(runResult([{ text: "for by grace you have been saved through faith and that not of yourselves it is the gift of god not of works", audioMs: 8000 }]), {
            llm: { provider: "openai", model: "gpt-4o-mini" },
            getApiKey: () => "test-key"
        })

        expect(mockDetectScripture).toHaveBeenCalledTimes(1)
        expect(replay.detections).toHaveLength(1)
        expect(replay.detections[0]).toMatchObject({ audioMs: 8000, reference: { bookNumber: 49, chapter: 2, verseStart: 8, source: "llm" } })
    })

    it("resolves a bare verse mention against the anchor passage", async () => {
        const replay = await replayDetection(runResult([{ text: "look with me at verse eleven", audioMs: 9000 }]), {
            anchor: { book: "Ephesians", bookNumber: 49, chapter: 2, verseStart: 8, verseEnd: 9 }
        })

        expect(replay.detections[0].reference).toMatchObject({ bookNumber: 49, chapter: 2, verseStart: 11 })
    })

    it("refuses a second replay while one is in flight, rather than losing the real clock to it", async () => {
        const realNow = Date.now
        let release = () => {}
        const parked = new Promise<void>((resolve) => (release = resolve))
        const inFlight = replayDetection(runResult([{ text: "john chapter three verse sixteen", audioMs: 1000 }]), { settle: () => parked })

        try {
            await expect(replayDetection(runResult([{ text: "romans chapter eight verse twenty eight", audioMs: 1000 }]))).rejects.toThrow(/concurrently/)
        } finally {
            release()
            await inFlight
        }

        expect(Date.now).toBe(realNow)
    })

    it("restores the real clock afterwards", async () => {
        const realNow = Date.now
        await replayDetection(runResult([{ text: "john chapter three verse sixteen", audioMs: 400000 }], 500000))

        expect(Date.now).toBe(realNow)
        expect(Math.abs(Date.now() - realNow())).toBeLessThan(1000)
    })
})

describe("bench/detection scoring", () => {
    it("scores recall, precision, latency and false positives per minute", async () => {
        const replay = replayOf([emission(4200, { bookNumber: 43, chapter: 3, verseStart: 16 }), emission(30000, { bookNumber: 1, chapter: 1, verseStart: 1 })])
        const score = scoreDetection(replay, [expected(), expected({ book: 45, chapter: 8, verseStart: 28, phraseEndMs: 20000 })])

        expect(score.matched).toBe(1)
        expect(score.recall).toBe(0.5)
        expect(score.precision).toBe(0.5)
        expect(score.missed).toEqual([expected({ book: 45, chapter: 8, verseStart: 28, phraseEndMs: 20000 })])
        expect(score.spurious).toHaveLength(1)
        expect(score.falsePositivesPerMinute).toBe(1)
        expect(score.latencyMs.p50).toBe(1200)
    })

    it("reports reference-free audio as false positives per minute, never as 0% recall", async () => {
        const score = scoreDetection(replayOf([emission(9000, { bookNumber: 4, chapter: 3, verseStart: 5 }), emission(21000, { bookNumber: 44, chapter: 15, verseStart: 1 })], 30000), [])

        expect(score.recall).toBeNull()
        expect(score.expectedCount).toBe(0)
        expect(score.falsePositives).toBe(2)
        expect(score.falsePositivesPerMinute).toBe(4)
        expect(score.precision).toBe(0)
        expect(score.latencyMs.n).toBe(0)
    })

    it("leaves precision undefined when nothing was detected at all", () => {
        const score = scoreDetection(replayOf([]), [])

        expect(score.precision).toBeNull()
        expect(score.recall).toBeNull()
        expect(score.falsePositivesPerMinute).toBe(0)
    })

    it("counts a re-detection of an already credited passage as a duplicate, not a false positive", () => {
        const score = scoreDetection(replayOf([emission(4000, { bookNumber: 43, chapter: 3, verseStart: 16 }), emission(200000, { bookNumber: 43, chapter: 3, verseStart: 16 })], 300000), [expected()])

        expect(score.matched).toBe(1)
        expect(score.duplicates).toBe(1)
        expect(score.falsePositives).toBe(0)
        expect(score.precision).toBe(1)
    })

    it("needs one detection per listed mention: a passage listed twice is not credited twice", () => {
        const score = scoreDetection(replayOf([emission(4000, { bookNumber: 43, chapter: 3, verseStart: 16 })], 300000), [expected(), expected({ phraseEndMs: 200000 })])

        expect(score.matched).toBe(1)
        expect(score.recall).toBe(0.5)
        expect(score.missed).toHaveLength(1)
        expect(score.duplicates).toBe(0)
    })

    it("refuses to credit a detection that fired before the phrase was spoken", () => {
        const score = scoreDetection(replayOf([emission(5000, { bookNumber: 43, chapter: 3, verseStart: 16 })]), [expected({ phraseEndMs: 20000 })])

        expect(score.matched).toBe(0)
        expect(score.falsePositives).toBe(1)
    })

    it("allows the hand-marking tolerance, with a negative latency", () => {
        const score = scoreDetection(replayOf([emission(19000, { bookNumber: 43, chapter: 3, verseStart: 16 })]), [expected({ phraseEndMs: 20000 })])

        expect(score.matched).toBe(1)
        expect(score.latencyMs.p50).toBe(-1000)
    })

    it("matches on the verse start and reports a lost range end separately", () => {
        const score = scoreDetection(replayOf([emission(21000, { bookNumber: 45, chapter: 8, verseStart: 28 })]), [expected({ book: 45, chapter: 8, verseStart: 28, verseEnd: 30, phraseEndMs: 20000 })])

        expect(score.matched).toBe(1)
        expect(score.recall).toBe(1)
        expect(score.rangeMismatches).toBe(1)
    })

    it("does not match a different verse of the right chapter", () => {
        const score = scoreDetection(replayOf([emission(4000, { bookNumber: 43, chapter: 3, verseStart: 17 })]), [expected()])

        expect(score.matched).toBe(0)
        expect(score.falsePositives).toBe(1)
        expect(score.recall).toBe(0)
    })

    it("pools recall over references and false positives over all audio, reference-free clips included", () => {
        const withReference = scoreDetection(replayOf([emission(4200, { bookNumber: 43, chapter: 3, verseStart: 16 })], 60000), [expected(), expected({ book: 45, chapter: 8, verseStart: 28, phraseEndMs: 20000 })])
        const referenceFree = scoreDetection(replayOf([emission(9000, { bookNumber: 4, chapter: 3, verseStart: 5 })], 60000), [])
        const summary = summarizeDetection([withReference, referenceFree])

        expect(summary.fixtures).toBe(2)
        expect(summary.referenceFixtures).toBe(1)
        expect(summary.expectedCount).toBe(2)
        expect(summary.recall).toBe(0.5)
        expect(summary.precision).toBe(0.5)
        expect(summary.falsePositivesPerMinute).toBe(0.5)
        expect(summary.latencyMs.n).toBe(1)
    })

    it("summarizes an entirely reference-free set without inventing a recall", () => {
        const summary = summarizeDetection([scoreDetection(replayOf([], 120000), []), scoreDetection(replayOf([emission(5000, { bookNumber: 44, chapter: 15, verseStart: 1 })], 120000), [])])

        expect(summary.recall).toBeNull()
        expect(summary.precision).toBe(0)
        expect(summary.falsePositivesPerMinute).toBe(0.25)
    })
})

describe("bench/detection end to end", () => {
    it("scores a spoken reference from an event log the way the report layer will", async () => {
        const replay = await replayDetection(
            runResult([
                { kind: "interim", text: "turn with me to ephesians", audioMs: 1200 },
                { kind: "segment", text: "turn with me to ephesians chapter two", audioMs: 2000, utteranceEnd: true },
                { kind: "segment", text: "verse eight", audioMs: 3600, utteranceEnd: true }
            ])
        )
        const score = scoreDetection(replay, [expected({ book: 49, chapter: 2, verseStart: 8, phrase: "ephesians chapter two verse eight", phraseEndMs: 3100 })])

        expect(score.recall).toBe(1)
        expect(score.matched).toBe(1)
        expect(score.latencyMs.max).toBe(500)
        // the first segment ends at the chapter, and the coordinator emits Ephesians 2:1 for it. The
        // manifest never asked for that verse, so it is a false positive - which is the honest score:
        // the operator really would see the wrong verse offered 1.6 s before the right one.
        expect(score.falsePositives).toBe(1)
        expect(score.spurious[0].reference).toMatchObject({ chapter: 2, verseStart: 1 })
    })
})

describe("scoreDetection late tolerance", () => {
    const reference: ExpectedReference = { book: 40, chapter: 6, verseStart: 33, phrase: "Matthew 6 33", phraseEndMs: 5000 }

    const replay = (audioMs: number): DetectionReplay => ({
        fixtureId: "t",
        variantId: "v",
        audioDurationMs: 3_600_000,
        detections: [{ audioMs, reference: { bookNumber: 40, chapter: 6, verseStart: 33, verseEnd: 33 } as DetectedReference }],
        segmentsReplayed: 1
    })

    it("does not credit a detection far outside the window", () => {
        // an unmarked re-mention 40 minutes later would otherwise score as a hit and inflate recall
        const score = scoreDetection(replay(2_400_000), [reference])
        expect(score.matches).toHaveLength(0)
        expect(score.missed).toHaveLength(1)
    })

    it("still treats that late detection as a repeat rather than a false positive", () => {
        // the manifest lists marked phrases, not every re-mention, so blaming precision for one
        // would make the number depend on how thorough the marking was
        const score = scoreDetection(replay(2_400_000), [reference])
        expect(score.spurious).toHaveLength(0)
        expect(score.duplicates).toBe(1)
    })

    it("credits a detection inside the window", () => {
        const score = scoreDetection(replay(12_000), [reference])
        expect(score.matches).toHaveLength(1)
        expect(score.matches[0].latencyMs).toBe(7000)
    })
})
