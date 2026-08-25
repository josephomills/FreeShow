import { describe, expect, it, vi } from "vitest"
import { NEMOTRON_CHUNK_SHIFT_MS, NEMOTRON_PRIMING_MS } from "../../setup/models/nemotronFiles"
import type { TranscriberSegment } from "../types"
import { NemotronStreamDriver } from "./streamDriver"

const SAMPLE_RATE = 16000
const pcm = (ms: number) => new Uint8Array(new Int16Array(Math.round((ms / 1000) * SAMPLE_RATE)).fill(1000).buffer)

/** A stream whose hypothesis is scripted verbatim, so a growing token can be reproduced exactly. */
function fake(steps: string[]) {
    const stream = {
        buffered: 0,
        consumed: 0,
        text: "",
        acceptWaveform(input: { samples: Float32Array }) {
            this.buffered += (input.samples.length / SAMPLE_RATE) * 1000
        }
    }
    let at = 0
    return {
        streams: [stream],
        sherpa: {
            OnlineRecognizer: class {
                createStream() {
                    return stream
                }
                isReady() {
                    return stream.buffered - stream.consumed >= (stream.consumed === 0 ? NEMOTRON_PRIMING_MS : NEMOTRON_CHUNK_SHIFT_MS)
                }
                decode() {
                    stream.consumed += stream.consumed === 0 ? NEMOTRON_PRIMING_MS : NEMOTRON_CHUNK_SHIFT_MS
                    if (at < steps.length) stream.text = steps[at++]
                }
                getResult() {
                    return { text: stream.text }
                }
                reset() {
                    stream.text = ""
                }
            },
            Vad: class {
                acceptWaveform() {}
                isDetected() {
                    return true
                }
                isEmpty() {
                    return true
                }
                pop() {}
            }
        }
    }
}

describe("a token that grows in place", () => {
    it("does not strand the characters added after it was committed", async () => {
        // reported live: the transcript showed "[MUSIC" with the closing bracket missing. The
        // hypothesis grows one BPE piece at a time, and a token with no trailing space is held
        // until it settles - so what happens to characters that arrive after that is the question.
        const { sherpa } = fake(["[MUS", "[MUSIC", "[MUSIC", "[MUSIC", "[MUSIC]", "[MUSIC] and"])
        const segments: TranscriberSegment[] = []

        const driver = new NemotronStreamDriver({
            paths: { encoder: "e", decoder: "d", joiner: "j", tokens: "t" },
            vadModelPath: "v",
            sherpa,
            onSegment: (segment) => segments.push(segment),
            onInterim: () => {},
            onError: vi.fn()
        })
        await driver.start()
        for (let sent = 0; sent < NEMOTRON_PRIMING_MS + NEMOTRON_CHUNK_SHIFT_MS * 7; sent += 100) driver.pushAudio(pcm(100))
        await driver.stop()

        const transcript = segments.map((s) => s.text).join(" ")
        console.log(`  emitted: ${JSON.stringify(segments.map((s) => s.text))}`)
        console.log(`  joined : ${JSON.stringify(transcript)}`)
        expect(transcript).toContain("[MUSIC]")
    })
})
