// AI BENCH - entry point.
//
// Two things live here. The pure-function tests always run: they are cheap and they guard the
// parts of the harness that could silently produce wrong numbers (a WAV parsed at the wrong
// offset, a pacer whose clock drifts). The engine runs are gated behind AI_BENCH=1 AND the model
// being present, following the RtmpStreamer.integration.test.ts precedent - `npm test` must stay
// fast and must not depend on a 650 MB download.
//
//   AI_BENCH=1 npx vitest run --config config/testing/vitest.config.ts src/electron/ai/speech/bench

import fs from "fs"
import os from "os"
import path from "path"
import { describe, expect, it } from "vitest"
import { availableVariants, benchEngineReady, resolveNemotronModelDir } from "./engines"
import { availableFixtures, listManifests, loadFixtureSet, resolveManifestDir, type Fixture, type FixtureSet } from "./fixtures"
import { align, normalizeForWer, vocabularyErrorRate } from "./align"
import { replayDetection, scoreDetection } from "./detection"
import { scoreRun } from "./metrics"
import { AudioClock, CHUNK_MS, CHUNK_SAMPLES, chunkToBytes, pace } from "./pacer"
import { renderConsoleReport, renderMarkdownDiff, toRunRecord, writeReport, type RunRecord } from "./report"
import { runFixture } from "./runner"
import { BENCH_SAMPLE_RATE, downmixToMono, floatToInt16, readWav16kMono, resampleLinear } from "./wav"

// PURE FUNCTIONS - always run

function writeTempWav(samples: Int16Array, sampleRate = BENCH_SAMPLE_RATE, channels = 1): string {
    const dataSize = samples.length * 2
    const buffer = Buffer.alloc(44 + dataSize)
    buffer.write("RIFF", 0, "ascii")
    buffer.writeUInt32LE(36 + dataSize, 4)
    buffer.write("WAVE", 8, "ascii")
    buffer.write("fmt ", 12, "ascii")
    buffer.writeUInt32LE(16, 16)
    buffer.writeUInt16LE(1, 20)
    buffer.writeUInt16LE(channels, 22)
    buffer.writeUInt32LE(sampleRate, 24)
    buffer.writeUInt32LE(sampleRate * 2 * channels, 28)
    buffer.writeUInt16LE(2 * channels, 32)
    buffer.writeUInt16LE(16, 34)
    buffer.write("data", 36, "ascii")
    buffer.writeUInt32LE(dataSize, 40)
    for (let i = 0; i < samples.length; i++) buffer.writeInt16LE(samples[i], 44 + i * 2)

    const file = path.join(fs.mkdtempSync(path.join(os.tmpdir(), "freeshow-bench-")), "test.wav")
    fs.writeFileSync(file, buffer)
    return file
}

describe("bench/wav", () => {
    it("round-trips 16 kHz mono PCM", () => {
        const samples = new Int16Array([0, 1000, -1000, 32767, -32768, 500])
        const wav = readWav16kMono(writeTempWav(samples))

        expect(wav.sampleRate).toBe(BENCH_SAMPLE_RATE)
        expect(wav.samples.length).toBe(samples.length)
        // int16 -> float -> int16 is lossy by at most one step
        for (let i = 0; i < samples.length; i++) expect(Math.abs(wav.samples[i] - samples[i])).toBeLessThanOrEqual(1)
    })

    it("finds the data chunk after an unrelated chunk", () => {
        // editors routinely prepend LIST/INFO; parsing at a fixed offset 44 would read metadata
        // as audio, which decodes to silence or noise rather than to an error
        const samples = new Int16Array([100, 200, 300, 400])
        const base = fs.readFileSync(writeTempWav(samples))

        const listChunk = Buffer.alloc(8 + 10)
        listChunk.write("LIST", 0, "ascii")
        listChunk.writeUInt32LE(10, 4)

        const spliced = Buffer.concat([base.subarray(0, 36), listChunk, base.subarray(36)])
        spliced.writeUInt32LE(spliced.length - 8, 4)

        const file = path.join(fs.mkdtempSync(path.join(os.tmpdir(), "freeshow-bench-")), "list.wav")
        fs.writeFileSync(file, spliced)

        const wav = readWav16kMono(file)
        expect(wav.samples.length).toBe(samples.length)
        expect(wav.samples[1]).toBeCloseTo(200, -1)
    })

    it("rejects a non-RIFF file with a specific reason", () => {
        const file = path.join(fs.mkdtempSync(path.join(os.tmpdir(), "freeshow-bench-")), "bad.wav")
        fs.writeFileSync(file, Buffer.from("this is not audio at all, not even close"))
        expect(() => readWav16kMono(file)).toThrow(/RIFF/)
    })

    it("downmixes stereo by averaging", () => {
        const interleaved = Float32Array.from([1, 0, 0.5, -0.5, -1, -1])
        expect(Array.from(downmixToMono(interleaved, 2))).toEqual([0.5, 0, -1])
    })

    it("resamples to the target length and passes 16k through untouched", () => {
        const input = new Float32Array(32000)
        expect(resampleLinear(input, 32000, 16000).length).toBe(16000)
        expect(resampleLinear(input, 16000, 16000)).toBe(input)
    })

    it("clamps out-of-range floats instead of wrapping", () => {
        // a float WAV can exceed +/-1; wrapping would turn a loud peak into a full-scale
        // opposite-sign spike, which is a click the decoder hears as a consonant
        const clamped = floatToInt16(Float32Array.from([1.5, -1.5]))
        expect(clamped[0]).toBe(32767)
        expect(clamped[1]).toBe(-32767)
    })
})

describe("bench/pacer", () => {
    it("feeds renderer-shaped 100 ms chunks and lands the clock on the audio duration", async () => {
        const samples = new Int16Array(BENCH_SAMPLE_RATE * 2) // 2s
        const clock = new AudioClock()
        const sizes: number[] = []

        const stats = await pace({ start: async () => {}, stop: async () => {}, pushAudio: (buffer) => sizes.push(buffer.byteLength) }, samples, clock, { mode: "max" })

        expect(sizes.length).toBe(20)
        expect(new Set(sizes)).toEqual(new Set([CHUNK_SAMPLES * 2]))
        expect(stats.audioPushedMs).toBe(2000)
        expect(CHUNK_MS).toBe(100)
    })

    it("advances the clock before pushAudio, so a synchronous emission is timestamped with its own audio", async () => {
        // the driver emits from inside pushAudio; if the clock advanced afterwards every segment
        // would be stamped 100 ms early and every latency number would be quietly wrong
        const samples = new Int16Array(BENCH_SAMPLE_RATE)
        const clock = new AudioClock()
        const seen: number[] = []

        await pace({ start: async () => {}, stop: async () => {}, pushAudio: () => seen.push(clock.audioPushedMs) }, samples, clock, { mode: "max" })

        expect(seen[0]).toBe(100)
        expect(seen[seen.length - 1]).toBe(1000)
    })

    it("emits a short trailing chunk rather than dropping it", () => {
        const samples = new Int16Array(CHUNK_SAMPLES + 400)
        const bytes = chunkToBytes(samples, CHUNK_SAMPLES, 400)
        expect(bytes.byteLength).toBe(800)
    })

    it("encodes Int16 little-endian, matching the renderer's worklet", () => {
        const bytes = chunkToBytes(Int16Array.from([-2, 258]), 0, 2)
        expect(Array.from(bytes)).toEqual([0xfe, 0xff, 0x02, 0x01])
    })
})

describe("bench/fixtures", () => {
    it("loads the committed smoke manifest", () => {
        const set = loadFixtureSet(path.join(resolveManifestDir(), "smoke.json"))
        expect(set.id).toBe("smoke-librispeech-v1")
        expect(set.fixtures.length).toBe(2)
        expect(set.hash).toMatch(/^[0-9a-f]{12}$/)
        expect(path.isAbsolute(set.fixtures[0].absolutePath)).toBe(true)
    })
})

describe("bench/align", () => {
    it("scores a clean match as zero", () => {
        const result = align(normalizeForWer("the sin which man thus punished"), normalizeForWer("The sin, which man thus punished."))
        expect(result.wer).toBe(0)
        expect(result.hits).toBe(6)
    })

    it("folds British spelling on both sides", () => {
        // scoring "dishonoured" vs "dishonored" as an error measures the corpus, not the engine
        expect(align(normalizeForWer("dishonoured honour centre"), normalizeForWer("dishonored honor center")).wer).toBe(0)
    })

    it("counts substitutions, deletions and insertions separately", () => {
        const result = align(["a", "b", "c", "d"], ["a", "x", "d", "e"])
        expect(result.substitutions + result.deletions + result.insertions).toBe(result.wer * 4)
        expect(result.refLength).toBe(4)
    })

    it("keeps single-letter words that the quote matcher would drop", () => {
        // baseTokens filters length-1 tokens because it is matching, not scoring; dropping "a"
        // here would forgive a real deletion
        expect(normalizeForWer("a lovely child")).toEqual(["a", "lovely", "child"])
    })

    it("prices vocabulary words separately from the headline WER", () => {
        const alignment = align(normalizeForWer("and ezra spoke to the people"), normalizeForWer("and extra spoke to the people"))
        const vocabulary = vocabularyErrorRate(alignment, new Set(["ezra"]))

        expect(alignment.wer).toBeCloseTo(1 / 6, 5) // one error in six words looks fine
        expect(vocabulary.rate).toBe(1) // but every name was wrong
        expect(vocabulary.missed).toEqual(["ezra"])
    })
})

describe("bench/metrics", () => {
    const run = (events: { kind: "segment" | "interim"; text: string; audioMs: number }[]) =>
        scoreRun({
            fixtureId: "t",
            fixturePath: "t.wav",
            variantId: "v",
            mode: "max",
            audioDurationMs: 5000,
            events: events.map((event) => ({ ...event, wallMs: 0 })),
            hypothesis: events
                .filter((event) => event.kind === "segment")
                .map((event) => event.text)
                .join(" "),
            pacer: { cpuMs: 500, audioPushedMs: 5000, wallMs: 0, pushBlockedMs: 500, pushBlockedP50: 0, pushBlockedP99: 0, pushBlockedMax: 0, driftMs: 0 },
            startupMs: 0,
            errors: [],
            platform: "test",
            arch: "test"
        })

    it("measures the gap between a word being visible and being committed", () => {
        const metrics = run([
            { kind: "interim", text: "after early", audioMs: 1700 },
            { kind: "segment", text: "after early", audioMs: 2900 }
        ])

        expect(metrics.commitLag?.words).toBe(2)
        expect(metrics.commitLag?.lag.p50).toBe(1200)
    })

    it("counts a word committed without ever being interim", () => {
        const metrics = run([{ kind: "segment", text: "sudden", audioMs: 1000 }])
        expect(metrics.commitLag?.neverInterim).toBe(1)
    })

    it("flags interim text that repeats already-committed words", () => {
        // seen live: a soft split re-hears the overlap, so the user sees the word twice even
        // though the seam stitch keeps the committed transcript clean
        const metrics = run([
            { kind: "segment", text: "dishonored bosom", audioMs: 9100 },
            { kind: "interim", text: "bosom", audioMs: 10300 }
        ])

        expect(metrics.interimEchoes).toHaveLength(1)
        expect(metrics.interimEchoes[0].text).toBe("bosom")
    })

    it("reports decode cost against audio duration", () => {
        expect(run([{ kind: "segment", text: "x", audioMs: 100 }]).decodeCostRatio).toBeCloseTo(0.1, 5)
    })
})

// REAL ENGINE - gated

// every manifest that resolves to audio on this machine: the committed public ones, plus any
// private set the operator generated into the fixture root (real sermons - see makeFixtures.js)
/**
 * A whole matrix in one process runs out of memory. Each loaded recognizer holds ~2 GB of native
 * ONNX state, and dropping the JS reference does not tell node's GC that 2 GB became free, so the
 * sessions pile up until the vitest worker is killed. AI_BENCH_SET and AI_BENCH_VARIANT let a
 * caller run one slice per process (scripts/ai/bench.sh does exactly that) - which is also the
 * only way the resource numbers are attributable to a single model.
 */
const setFilter = process.env.AI_BENCH_SET
const variantFilter = process.env.AI_BENCH_VARIANT

const runnableSets: { set: FixtureSet; fixtures: Fixture[] }[] = listManifests()
    .map((manifestPath) => {
        const set = loadFixtureSet(manifestPath)
        return { set, fixtures: availableFixtures(set) }
    })
    .filter((entry) => entry.fixtures.length > 0 && (!setFilter || entry.set.id === setFilter))

const canRun = !!process.env.AI_BENCH && benchEngineReady("nemotron") && runnableSets.length > 0
const describeIfEngine = canRun ? describe : describe.skip

if (process.env.AI_BENCH && !canRun) {
    console.warn(`[bench] skipped: model dir ${resolveNemotronModelDir()}, ${runnableSets.length} runnable fixture set(s)`)
}

describeIfEngine("bench/nemotron (real model)", () => {
    for (const { set, fixtures } of runnableSets) {
        it(`scores ${set.id} (${fixtures.length} fixtures)`, async () => {
            const variants = availableVariants().filter((variant) => !variantFilter || variant.id === variantFilter)
            const records: RunRecord[] = []
            const stamp = Number(process.env.AI_BENCH_STAMP) || 0

            for (const fixture of fixtures) {
                for (const variant of variants) {
                    const result = await runFixture({ fixtureId: fixture.id, fixturePath: fixture.absolutePath, variant, mode: "max" })
                    const metrics = scoreRun(result, fixture.transcript)

                    // The product metric. Replayed from the event log this decode already produced,
                    // so it costs no audio and no engine - but it must be awaited here in the loop
                    // rather than gathered up afterwards: replayDetection drives the coordinator off
                    // a patched global Date.now and refuses to run while another replay holds it.
                    const detection = scoreDetection(await replayDetection(result), fixture.expected ?? [])
                    records.push(toRunRecord(result, metrics, set, stamp, detection))

                    // the invariant every latency number rests on
                    const audioTimes = result.events.map((event) => event.audioMs)
                    expect(audioTimes).toEqual([...audioTimes].sort((a, b) => a - b))
                    expect(result.errors).toEqual([])

                    // An empty hypothesis is a legitimate outcome, not a failure. A fixed-offset
                    // excerpt sometimes lands on worship rather than preaching - one 120s clip's
                    // reference transcript is "Oh, oh, oh" forty-four times - and an engine that
                    // returns nothing there is arguably behaving correctly. Failing the slice on it
                    // discards the whole fixture set for every variant.
                    if (!result.hypothesis) console.warn(`  [${fixture.id} / ${variant.id}] empty transcript - check whether this excerpt is speech`)

                    // named while the fixture is fresh in the log; the pooled recall in the report
                    // says how many were missed but never which phrase the operator would have lost
                    for (const missed of detection.missed) console.warn(`  [${fixture.id} / ${variant.id}] missed "${missed.phrase}" at ${missed.phraseEndMs}ms`)
                }
            }

            console.log(`\n${renderConsoleReport(records)}`)
            console.log(`\n${renderMarkdownDiff(records, variants[0].id)}`)
            console.log(`\nreport -> ${writeReport(records)}`)
        }, 3600000)
    }
})
