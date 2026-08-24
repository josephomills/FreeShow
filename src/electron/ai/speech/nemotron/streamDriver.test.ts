import { describe, expect, it } from "vitest"

import { NEMOTRON_CHUNK_SHIFT_MS, NEMOTRON_PRIMING_MS } from "../../setup/models/nemotronFiles"
import type { TranscriberSegment } from "../types"
import { NemotronStreamDriver } from "./streamDriver"

const SAMPLE_RATE = 16000
const CHUNK_MS = 100

function pcm(ms: number, value = 1000): Uint8Array {
    return new Uint8Array(new Int16Array(Math.round((ms / 1000) * SAMPLE_RATE)).fill(value).buffer)
}

/**
 * A fake sherpa that models the ENCODER GRID, which is the whole point: the real model runs its
 * first encoder step once 1300ms of audio is buffered and one every 1120ms after that, and the
 * driver's emission timing is a consequence of that cadence. The old fake in driver.test.ts drives
 * decoding purely from getResult and models nothing about it, so it cannot catch a regression in
 * any of the behaviour this file cares about.
 *
 * Words are handed out one per encoder step, so a test can reason in grid units.
 */
interface FakeControls {
    detected: boolean
    closedQueue: number
    /** Emitted one per encoder step. Text accumulates - greedy RNN-T never retracts. */
    words: string[]
    /** Set to make getResult return something that is NOT a prefix-extension, once. */
    hostileRevision?: string
}

class FakeStream {
    /** How many times the decoder state was cleared - the observable effect of breaking a loop. */
    resets = 0
    buffered = 0
    consumed = 0
    pendingSteps = 0
    text = ""
    startTime = 0
    acceptedCalls = 0
}

function makeSherpa(controls: FakeControls) {
    const streams: FakeStream[] = []
    let nextWord = 0

    class OnlineRecognizer {
        createStream() {
            const stream = new FakeStream()
            streams.push(stream)
            return stream
        }
        isReady(stream: FakeStream) {
            // first step needs the priming window; every step after it needs one chunk shift
            const needed = stream.consumed === 0 ? NEMOTRON_PRIMING_MS : NEMOTRON_CHUNK_SHIFT_MS
            return stream.buffered - stream.consumed >= needed
        }
        decode(stream: FakeStream) {
            stream.consumed += stream.consumed === 0 ? NEMOTRON_PRIMING_MS : NEMOTRON_CHUNK_SHIFT_MS
            stream.pendingSteps++
            if (nextWord < controls.words.length) {
                stream.text = stream.text ? `${stream.text} ${controls.words[nextWord]}` : controls.words[nextWord]
                nextWord++
            }
        }
        getResult(stream: FakeStream) {
            if (controls.hostileRevision) {
                stream.text = controls.hostileRevision
                controls.hostileRevision = undefined
            }
            return { text: stream.text, start_time: stream.startTime }
        }
        reset(stream: FakeStream) {
            stream.resets++
            stream.text = ""
            stream.startTime = stream.consumed / 1000
        }
    }

    class Vad {
        acceptWaveform() {}
        isDetected() {
            return controls.detected
        }
        isEmpty() {
            return controls.closedQueue <= 0
        }
        pop() {
            controls.closedQueue--
        }
    }

    // the driver calls stream.acceptWaveform({ sampleRate, samples })
    ;(FakeStream.prototype as unknown as { acceptWaveform: (input: { samples: Float32Array }) => void }).acceptWaveform = function (this: FakeStream, input) {
        this.acceptedCalls++
        this.buffered += (input.samples.length / SAMPLE_RATE) * 1000
    }

    return { sherpa: { OnlineRecognizer, Vad }, streams }
}

interface Harness {
    driver: NemotronStreamDriver
    segments: TranscriberSegment[]
    interims: string[]
    errors: string[]
    streams: FakeStream[]
    controls: FakeControls
    push: (ms?: number) => void
}

async function harness(overrides: Partial<FakeControls> & { resetIntervalMs?: number } = {}): Promise<Harness> {
    const { resetIntervalMs, ...rest } = overrides
    const controls: FakeControls = { detected: true, closedQueue: 0, words: [], ...rest }
    const { sherpa, streams } = makeSherpa(controls)

    const segments: TranscriberSegment[] = []
    const interims: string[] = []
    const errors: string[] = []

    const driver = new NemotronStreamDriver({
        paths: { encoder: "e", decoder: "d", joiner: "j", tokens: "t" },
        vadModelPath: "v",
        language: "en",
        resetIntervalMs,
        sherpa,
        onSegment: (segment) => segments.push(segment),
        onInterim: (text) => interims.push(text),
        onError: (message) => errors.push(message)
    })
    await driver.start()

    return {
        driver,
        segments,
        interims,
        errors,
        streams,
        controls,
        push: (ms = CHUNK_MS) => {
            for (let sent = 0; sent < ms; sent += CHUNK_MS) driver.pushAudio(pcm(CHUNK_MS))
        }
    }
}

const textOf = (segments: TranscriberSegment[]) =>
    segments
        .map((segment) => segment.text)
        .filter(Boolean)
        .join(" ")

describe("NemotronStreamDriver", () => {
    it("opens exactly one stream for the whole session", async () => {
        const h = await harness({ words: ["a", "b", "c", "d", "e"] })
        h.push(8000)
        h.controls.closedQueue = 1
        h.controls.detected = false
        h.push(2000)
        h.push(4000)

        // recreating the stream is precisely the defect this driver exists to fix: it discards the
        // encoder cache and re-pays the 1300ms priming
        expect(h.streams).toHaveLength(1)
    })

    it("feeds every push to the recognizer, including silence", async () => {
        // the cache-warmth invariant. Gating on vad.isDetected() would look like an optimisation
        // and would quietly re-introduce a priming cost at each utterance start
        const h = await harness({ detected: false, words: [] })
        h.push(3000)

        expect(h.streams[0].acceptedCalls).toBe(30)
    })

    it("emits a word on the step that produces it, with no agreement delay", async () => {
        const h = await harness({ words: ["alpha", "bravo", "charlie"] })

        // priming + one shift = two steps = two words; the trailing one is held back
        h.push(NEMOTRON_PRIMING_MS + NEMOTRON_CHUNK_SHIFT_MS)

        expect(textOf(h.segments)).toBe("alpha")
        expect(h.interims[h.interims.length - 1]).toBe("bravo")
    })

    it("never commits a partial trailing token, however long the hypothesis sits still", async () => {
        // A greedy RNN-T grows the trailing token IN PLACE, one BPE piece at a time - "[MUS" to
        // "[MUSIC" to "[MUSIC]". This driver used to commit it once the hypothesis had been static
        // for an encoder step, on the theory that it had settled. Live transcripts showed what that
        // really means: "[MUSIC" on screen and "]" stranded after it, and "chapter" truncated to
        // "cha", which took a scripture reference with it.
        const h = await harness({ words: ["alpha", "bravo"] })
        h.push(NEMOTRON_PRIMING_MS + NEMOTRON_CHUNK_SHIFT_MS)
        expect(textOf(h.segments)).toBe("alpha")

        h.push(NEMOTRON_CHUNK_SHIFT_MS * 4)
        expect(textOf(h.segments)).toBe("alpha")

        // it commits when the utterance closes, which the VAD does within a second of a pause
        h.controls.detected = false
        h.controls.closedQueue = 1
        h.push(NEMOTRON_CHUNK_SHIFT_MS + 300)
        expect(textOf(h.segments)).toBe("alpha bravo")
    })

    it("closes an utterance only after a full chunk shift of real audio past the VAD close", async () => {
        // the shipped batch driver defers 500ms, which is LESS than one 1120ms step - under a
        // persistent stream that closes before the final word's own chunk has run
        const h = await harness({ words: ["alpha", "bravo", "charlie"] })
        h.push(NEMOTRON_PRIMING_MS)

        h.controls.detected = false
        h.controls.closedQueue = 1
        h.push(500)
        expect(h.segments.some((segment) => segment.utteranceEnd)).toBe(false)

        h.push(NEMOTRON_CHUNK_SHIFT_MS)
        expect(h.segments.some((segment) => segment.utteranceEnd)).toBe(true)
    })

    it("clears the hypothesis on an utterance boundary so the next one carries no prefix", async () => {
        const h = await harness({ words: ["alpha", "bravo"] })
        h.push(NEMOTRON_PRIMING_MS + NEMOTRON_CHUNK_SHIFT_MS)

        h.controls.detected = false
        h.controls.closedQueue = 1
        h.push(NEMOTRON_CHUNK_SHIFT_MS + 200)

        const beforeCount = h.segments.length
        h.controls.detected = true
        h.controls.words.push("delta", "echo")
        h.push(NEMOTRON_PRIMING_MS + NEMOTRON_CHUNK_SHIFT_MS * 2)

        const after = h.segments.slice(beforeCount)
        expect(textOf(after)).not.toContain("alpha")
    })

    it("never emits the same text twice across a boundary", async () => {
        const h = await harness({ words: ["alpha", "bravo", "charlie", "delta"] })
        h.push(NEMOTRON_PRIMING_MS + NEMOTRON_CHUNK_SHIFT_MS * 2)
        h.controls.detected = false
        h.controls.closedQueue = 1
        h.push(NEMOTRON_CHUNK_SHIFT_MS + 200)
        h.controls.detected = true
        h.push(NEMOTRON_PRIMING_MS + NEMOTRON_CHUNK_SHIFT_MS * 2)

        const words = textOf(h.segments).split(/\s+/).filter(Boolean)
        expect(new Set(words).size).toBe(words.length)
    })

    it("keeps segment times monotonic", async () => {
        const h = await harness({ words: ["alpha", "bravo", "charlie"] })
        h.push(NEMOTRON_PRIMING_MS + NEMOTRON_CHUNK_SHIFT_MS * 2)
        h.controls.closedQueue = 1
        h.controls.detected = false
        h.push(NEMOTRON_CHUNK_SHIFT_MS + 200)

        let previous = -1
        for (const segment of h.segments) {
            expect(segment.startMs).toBeGreaterThanOrEqual(previous)
            expect(segment.endMs).toBeGreaterThanOrEqual(segment.startMs)
            previous = segment.endMs
        }
    })

    it("does not duplicate emitted text when the hypothesis is revised instead of extended", async () => {
        // cannot happen with greedy RNN-T today, but sherpa's homophone replacer and rule_fsts
        // post-processors would do exactly this, and silently duplicating text is the bad outcome
        const h = await harness({ words: ["alpha", "bravo", "charlie"] })
        h.push(NEMOTRON_PRIMING_MS + NEMOTRON_CHUNK_SHIFT_MS)
        expect(textOf(h.segments)).toBe("alpha")

        h.controls.hostileRevision = "completely different words here"
        h.push(NEMOTRON_CHUNK_SHIFT_MS * 2)

        expect(textOf(h.segments).startsWith("alpha")).toBe(true)
        expect(textOf(h.segments)).not.toContain("alpha alpha")
    })

    it("flushes audio past the last encoder step on stop", async () => {
        // without this the final words of a session are simply never decoded - the bench caught
        // it as a dropped "brothels" at the end of a 6.6s clip
        const h = await harness({ words: ["alpha", "bravo"] })
        h.push(NEMOTRON_PRIMING_MS)
        expect(textOf(h.segments)).toBe("")

        await h.driver.stop()
        expect(textOf(h.segments)).toContain("alpha")
    })

    it("is a no-op after stop", async () => {
        const h = await harness({ words: ["alpha"] })
        await h.driver.stop()
        const before = h.segments.length

        h.push(5000)
        expect(h.segments).toHaveLength(before)
        expect(h.errors).toEqual([])
    })

    it("reports a recognizer failure through onError rather than throwing", async () => {
        const h = await harness({ words: [] })
        h.streams[0].acceptWaveform = () => {
            throw new Error("native decode blew up")
        }

        expect(() => h.push(200)).not.toThrow()
        expect(h.errors[0]).toContain("native decode blew up")
    })

    it("clears the decoder at an utterance boundary", async () => {
        // Carrying the RNN-T predictor's state across boundaries measurably improved transcription
        // on 120 s fixtures, so this driver did that for a while. Live use over a full service
        // showed why it was wrong: the predictor's own output is its next input, and given a long
        // enough unbroken run it locks into a cycle and fills the transcript with one phrase. The
        // boundary was bounding that. Short fixtures cannot show a failure that needs minutes of
        // continuous decoding to appear.
        const h = await harness({ words: ["alpha", "bravo", "charlie", "delta"] })

        h.push(NEMOTRON_PRIMING_MS + NEMOTRON_CHUNK_SHIFT_MS * 2)
        h.controls.detected = false
        h.controls.closedQueue = 1
        h.push(NEMOTRON_CHUNK_SHIFT_MS + 300)

        expect(h.streams[0].text).toBe("")
    })

    it("still never repeats committed text without a reset to clear it", async () => {
        const h = await harness({ words: ["alpha", "bravo", "charlie", "delta", "echo"] })
        h.push(NEMOTRON_PRIMING_MS + NEMOTRON_CHUNK_SHIFT_MS * 2)
        h.controls.detected = false
        h.controls.closedQueue = 1
        h.push(NEMOTRON_CHUNK_SHIFT_MS + 300)
        h.controls.detected = true
        h.push(NEMOTRON_CHUNK_SHIFT_MS * 3)

        const words = textOf(h.segments).split(/\s+/).filter(Boolean)
        expect(new Set(words).size).toBe(words.length)
    })

    it("can be asked to hold its state across a boundary, for measurement", async () => {
        // the switch stays so the bench can price what the boundary reset costs; production does
        // not set it
        const h = await harness({ words: ["alpha", "bravo"], resetIntervalMs: 600_000 })
        h.push(NEMOTRON_PRIMING_MS + NEMOTRON_CHUNK_SHIFT_MS)
        h.controls.detected = false
        h.controls.closedQueue = 1
        h.push(NEMOTRON_CHUNK_SHIFT_MS + 300)

        expect(h.streams[0].text).not.toBe("")
    })

    it("breaks a decoder loop instead of transcribing it", async () => {
        // seen live: the transcript filled with "and he saith the LORD" over and over. The
        // predictor's own output is its next input, so nothing in the audio pulls it out - only
        // clearing the decoder state does.
        // five repeats of a three-word phrase - well past rhetorical repetition, which a preacher
        // does deliberately and which must not clear the decoder mid-sentence
        const cycle = ["saith", "the", "lord"]
        const words = [...cycle, ...cycle, ...cycle, ...cycle, ...cycle]
        const h = await harness({ words })
        h.push(NEMOTRON_PRIMING_MS + NEMOTRON_CHUNK_SHIFT_MS * (words.length + 1))

        // clearing the decoder state is what actually stops it - a loop can only be recognised once
        // it has repeated, so some of it always escapes first. Catching it after a handful of
        // repeats instead of twenty is the whole gain.
        expect(h.streams[0].resets).toBeGreaterThan(0)
        const emitted = textOf(h.segments).split(/\s+/).filter(Boolean)
        expect(emitted.length).toBeLessThan(words.length)
    })

    it("keeps the first occurrence of the repeated phrase", async () => {
        const h = await harness({ words: ["holy", "is", "he", "amen", "amen", "amen", "amen", "amen", "amen", "amen", "amen"] })
        h.push(NEMOTRON_PRIMING_MS + NEMOTRON_CHUNK_SHIFT_MS * 12)

        expect(textOf(h.segments)).toContain("holy is he")
    })

    it("marks a music label so detection never sees it", async () => {
        // seen live during a worship set: the engine labelled the audio rather than inventing words
        // for it, and those labels arrived at scripture detection as ordinary speech
        const h = await harness({ words: ["[MUSIC", "PLAYING]", "welcome"] })
        h.push(NEMOTRON_PRIMING_MS + NEMOTRON_CHUNK_SHIFT_MS * 3)
        h.controls.detected = false
        h.controls.closedQueue = 1
        h.push(NEMOTRON_CHUNK_SHIFT_MS + 300)

        const labelled = h.segments.filter((segment) => segment.text.includes("MUSIC"))
        expect(labelled.length).toBeGreaterThan(0)
        expect(labelled.every((segment) => segment.music)).toBe(true)
    })

    it("does not mark ordinary speech as music", async () => {
        const h = await harness({ words: ["turn", "to", "ephesians"] })
        h.push(NEMOTRON_PRIMING_MS + NEMOTRON_CHUNK_SHIFT_MS * 3)
        h.controls.detected = false
        h.controls.closedQueue = 1
        h.push(NEMOTRON_CHUNK_SHIFT_MS + 300)

        expect(h.segments.some((segment) => segment.music)).toBe(false)
    })
})
