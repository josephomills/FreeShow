// AI BENCH - report guardrails and formatting. Pure functions plus one temp-dir write, so this
// always runs; nothing here needs a model or a fixture.

import fs from "fs"
import os from "os"
import path from "path"
import { describe, expect, it } from "vitest"
import type { DetectedReference } from "../../../../types/ai/AiScripture"
import { scoreDetection, type DetectionEmission, type DetectionScore } from "./detection"
import type { ExpectedReference } from "./fixtures"
import type { RunMetrics } from "./metrics"
import type { PaceMode } from "./pacer"
import { assertComparable, buildReport, checkComparable, formatCi, formatMs, formatPercent, formatRatio, renderConsoleReport, renderMarkdownDiff, renderTable, resolveReportDir, summarizeByVariant, summarizeVariant, writeReport, type RunRecord } from "./report"
import { describe as describeDistribution } from "./stats"

interface Sample {
    fixtureId?: string
    variantId?: string
    hash?: string
    platform?: string
    arch?: string
    mode?: PaceMode
    resampledFrom?: number
    audioDurationMs?: number
    lag?: number[]
    wer?: number
    refLength?: number
    vocabRate?: number
    echoes?: number
    decodeCost?: number
    startupMs?: number
    errors?: string[]
    detection?: DetectionSample
}

interface DetectionSample {
    /** References the manifest lists. 0 is the common case - most clips ask for no verse at all. */
    expected?: number
    /** How many of them the replay found. Defaults to all of them. */
    found?: number
    /** Detections of a passage nobody listed. */
    falsePositives?: number
    latencyMs?: number
}

/**
 * Scored by the real scoreDetection rather than assembled by hand, so a test can never assert a
 * combination of counts that detection.ts would not produce.
 */
function makeDetection(sample: DetectionSample, fixtureId: string, variantId: string, audioDurationMs: number): DetectionScore {
    // one verse per reference, so a detection can only ever be credited to the mention it belongs to
    const expected: ExpectedReference[] = Array.from({ length: sample.expected ?? 0 }, (_, index) => ({ book: 40, chapter: 6, verseStart: 33 + index, phrase: `matthew six ${33 + index}`, phraseEndMs: 5000 * (index + 1) }))
    const latencyMs = sample.latencyMs ?? 1200
    const detected = (bookNumber: number, chapter: number, verseStart: number) => ({ bookNumber, chapter, verseStart, verseEnd: verseStart }) as DetectedReference

    const detections: DetectionEmission[] = expected.slice(0, sample.found ?? expected.length).map((reference) => ({ audioMs: reference.phraseEndMs + latencyMs, reference: detected(reference.book, reference.chapter, reference.verseStart) }))
    // Genesis 1:N - a book no sample lists, so these are spurious rather than repeats
    for (let index = 0; index < (sample.falsePositives ?? 0); index++) detections.push({ audioMs: 1000 * (index + 1), reference: detected(1, 1, index + 1) })

    return scoreDetection({ fixtureId, variantId, audioDurationMs, segmentsFed: detections.length, detections, statuses: [] }, expected)
}

function makeRecord(sample: Sample = {}): RunRecord {
    const audioDurationMs = sample.audioDurationMs ?? 60000
    const lag = sample.lag ?? [100, 200, 300]
    const fixtureId = sample.fixtureId ?? "fixture-a"
    const variantId = sample.variantId ?? "batch"

    const metrics: RunMetrics = {
        fixtureId,
        variantId,
        audioDurationMs,
        commitLag: { lag: describeDistribution(lag), neverInterim: 0, words: lag.length },
        interimEchoes: Array.from({ length: sample.echoes ?? 0 }, (_, index) => ({ audioMs: index * 1000, text: "echo" })),
        decodeCostRatio: sample.decodeCost ?? 0.25,
        startupMs: sample.startupMs ?? 1200,
        errors: sample.errors ?? []
    }

    if (sample.wer !== undefined) {
        const refLength = sample.refLength ?? 100
        metrics.wer = {
            wer: sample.wer,
            substitutions: 1,
            deletions: 0,
            insertions: 0,
            refLength,
            alignment: { pairs: [{ op: "match", refIndex: 0, hypIndex: 0, refToken: "and", hypToken: "and" }], substitutions: 1, deletions: 0, insertions: 0, hits: refLength - 1, refLength, wer: sample.wer }
        }
    }
    if (sample.vocabRate !== undefined) metrics.vocabulary = { errors: 1, total: 10, rate: sample.vocabRate, missed: ["ezra"] }

    return {
        fixtureSetId: "smoke",
        fixtureSetHash: sample.hash ?? "aaaabbbbcccc",
        fixtureId,
        variantId,
        mode: sample.mode ?? "max",
        platform: sample.platform ?? "darwin",
        arch: sample.arch ?? "arm64",
        timestampMs: 1700000000000,
        ...(sample.resampledFrom ? { resampledFrom: sample.resampledFrom } : {}),
        ...(sample.detection ? { detection: makeDetection(sample.detection, fixtureId, variantId, audioDurationMs) } : {}),
        metrics
    }
}

/** Four fixtures x two variants, the stream variant slightly worse on WER. */
function abRecords(): RunRecord[] {
    const werByVariant: Record<string, number[]> = { batch: [0.1, 0.12, 0.09, 0.11], stream: [0.14, 0.15, 0.13, 0.16] }
    const records: RunRecord[] = []

    for (const variantId of ["batch", "stream"]) {
        werByVariant[variantId].forEach((wer, index) => {
            records.push(makeRecord({ variantId, fixtureId: `fixture-${index}`, wer, lag: variantId === "batch" ? [800, 900, 1000] : [200, 250, 300], decodeCost: variantId === "batch" ? 0.2 : 0.3, echoes: variantId === "batch" ? 0 : 2 }))
        })
    }
    return records
}

describe("bench/report guardrail: comparability", () => {
    it("refuses records from different fixture-set hashes and says why", () => {
        const records = [makeRecord({ hash: "aaaabbbbcccc" }), makeRecord({ variantId: "stream", hash: "ddddeeeeffff" })]

        expect(
            checkComparable(records)
                .filter((issue) => issue.blocking)
                .map((issue) => issue.field)
        ).toEqual(["fixtureSet"])
        expect(() => assertComparable(records)).toThrow(/fixture sets differ/)
        expect(() => renderConsoleReport(records)).toThrow(/reference transcripts/)
        expect(() => renderMarkdownDiff(records, "batch")).toThrow(/fixture sets differ/)
    })

    it("refuses records from different machines", () => {
        const records = [makeRecord(), makeRecord({ variantId: "stream", platform: "win32" })]
        expect(() => renderConsoleReport(records)).toThrow(/different machines \(darwin\/arm64, win32\/arm64\)/)

        const architectures = [makeRecord(), makeRecord({ variantId: "stream", arch: "x64" })]
        expect(() => renderConsoleReport(architectures)).toThrow(/different machines/)
    })

    it("refuses records paced in different modes", () => {
        const records = [makeRecord({ mode: "max" }), makeRecord({ variantId: "stream", mode: "rt" })]
        expect(() => renderMarkdownDiff(records, "batch")).toThrow(/pacing modes differ \(max, rt\)/)
    })

    it("renders a like-for-like set without complaint", () => {
        expect(checkComparable(abRecords())).toEqual([])
        expect(() => renderConsoleReport(abRecords())).not.toThrow()
    })

    it("warns about resampled audio instead of refusing it", () => {
        // a resampled fixture is still a fair A/B, it just is not the engine's native-rate WER
        const records = [makeRecord({ wer: 0.1 }), makeRecord({ variantId: "stream", wer: 0.2, resampledFrom: 44100 })]

        expect(checkComparable(records).every((issue) => !issue.blocking)).toBe(true)
        expect(renderConsoleReport(records)).toMatch(/! resample: resampled to 16 kHz: fixture-a@44100Hz/)
        expect(renderMarkdownDiff(records, "batch")).toMatch(/> \*\*Warning\*\* resample:/)
    })

    it("keeps a blocking reason in the written report, marked", () => {
        // writeReport never throws: a report of a broken comparison must still be inspectable,
        // but it may not look clean
        const report = buildReport([makeRecord(), makeRecord({ variantId: "stream", mode: "rt" })])
        expect(report.warnings.some((warning) => warning.startsWith("BLOCKING mode:"))).toBe(true)
        expect(report.modes).toEqual(["max", "rt"])
    })
})

describe("bench/report guardrail: n and interval", () => {
    it("prints n=1 rather than an interval invented from one run", () => {
        const summary = summarizeVariant("batch", [makeRecord({ wer: 0.1 })])

        expect(summary.wer?.n).toBe(1)
        expect(Number.isFinite(summary.wer!.low)).toBe(false)
        expect(formatCi(summary.wer, formatPercent)).toBe("n=1")
        expect(renderConsoleReport([makeRecord({ wer: 0.1 })])).toContain("n=1")
    })

    it("prints n and a bootstrap interval that brackets the mean", () => {
        const summary = summarizeVariant(
            "batch",
            abRecords().filter((record) => record.variantId === "batch")
        )

        expect(summary.wer?.n).toBe(4)
        expect(summary.wer!.low).toBeLessThanOrEqual(summary.wer!.value)
        expect(summary.wer!.high).toBeGreaterThanOrEqual(summary.wer!.value)
        expect(summary.wer!.low).toBeGreaterThanOrEqual(0.09)
        expect(summary.wer!.high).toBeLessThanOrEqual(0.12)
    })

    it("carries n and an interval next to every rate in the console table", () => {
        const table = renderConsoleReport(abRecords())

        expect(table).toContain("WER 95% CI")
        expect(table).toContain("echo CI")
        expect(table).toMatch(/all fixtures \| n=4/)
    })

    it("marks an overlapping delta as noise", () => {
        // same distribution shuffled between the two variants: whatever the means do, the
        // intervals sit on top of each other and the delta is not a finding
        const values = [0.1, 0.2, 0.3, 0.4]
        const records = values.flatMap((wer, index) => [makeRecord({ variantId: "batch", fixtureId: `f${index}`, wer }), makeRecord({ variantId: "stream", fixtureId: `f${index}`, wer: values[values.length - 1 - index] })])

        expect(renderMarkdownDiff(records, "batch")).toContain("within noise")
    })

    it("does not call it noise when the intervals are clear of each other", () => {
        const werRow = renderMarkdownDiff(abRecords(), "batch")
            .split("\n")
            .find((line) => line.startsWith("| WER |"))!

        expect(werRow).not.toContain("within noise")
    })
})

describe("bench/report formatting", () => {
    it("formats ms as integers, ratios to 3dp and WER as a percentage", () => {
        expect(formatMs(1234.6)).toBe("1235")
        expect(formatRatio(0.31789)).toBe("0.318")
        expect(formatPercent(0.1234)).toBe("12.3%")
        expect(formatMs(Number.NaN)).toBe("-")
        expect(formatRatio(Number.NaN)).toBe("-")
    })

    it("aligns every column to the widest cell", () => {
        const table = renderTable(
            [
                { header: "variant", align: "left" },
                { header: "WER", align: "right" }
            ],
            [
                ["a-very-long-variant-id", "9.9%"],
                ["b", "12.3%"]
            ]
        )
        const lines = table.split("\n")

        expect(lines[0]).toBe("  variant                   WER")
        expect(lines[2]).toBe("  a-very-long-variant-id   9.9%")
        expect(lines[3]).toBe("  b                       12.3%")
    })

    it("stays inside 120 columns with realistic variant ids", () => {
        const records = ["nemotron-batch-en", "nemotron-stream-en", "nemotron-stream-en-ctc"].flatMap((variantId) => [0, 1].map((index) => makeRecord({ variantId, fixtureId: `fixture-${index}`, wer: 0.12, vocabRate: 0.3, echoes: 3 })))

        for (const line of renderConsoleReport(records).split("\n")) expect(line.length).toBeLessThanOrEqual(120)
    })

    it("groups one row per variant under each fixture", () => {
        const lines = renderConsoleReport(abRecords()).split("\n")

        expect(lines.filter((line) => line.startsWith("=== fixture-"))).toHaveLength(4)
        expect(lines.filter((line) => line.trimStart().startsWith("batch "))).toHaveLength(5) // 4 fixtures + the pooled table
    })

    it("shows the reference word count next to the fixture, not in a column", () => {
        expect(renderConsoleReport([makeRecord({ wer: 0.1, refLength: 148, audioDurationMs: 62400 })])).toContain("=== fixture-a | 62.4s | 148 ref words ===")
    })

    it("omits the vocabulary columns when nothing scored a vocabulary", () => {
        expect(renderConsoleReport([makeRecord({ wer: 0.1 })])).not.toContain("vocab")
        expect(renderConsoleReport([makeRecord({ wer: 0.1, vocabRate: 0.25 })])).toContain("vocab CI")
    })

    it("renders identically twice - nothing reads the clock", () => {
        expect(renderConsoleReport(abRecords())).toBe(renderConsoleReport(abRecords()))
        expect(renderMarkdownDiff(abRecords(), "batch")).toBe(renderMarkdownDiff(abRecords(), "batch"))
    })
})

describe("bench/report markdown diff", () => {
    it("labels the sign that means better on every metric", () => {
        const markdown = renderMarkdownDiff(abRecords(), "batch")
        const row = (label: string) => markdown.split("\n").find((line) => line.startsWith(`| ${label} |`))!

        expect(markdown).toContain("| metric | better when | `batch` (baseline) | `stream` | delta |")
        // stream is 4 points worse on WER and 650 ms better on lag - the words carry the sign
        expect(row("WER")).toContain("+4.0 pp **worse**")
        expect(row("WER")).toContain("lower")
        expect(row("commit lag mean (ms)")).toContain("-650 **better**")
        expect(row("decode cost (x audio)")).toContain("+0.100 **worse**")
    })

    it("states n per variant and next to every rate", () => {
        const markdown = renderMarkdownDiff(abRecords(), "batch")

        expect(markdown).toContain("Runs per variant: `batch` n=4, `stream` n=4.")
        expect(markdown).toMatch(/\| WER \| lower \| 10\.5% \(n=4, 95% CI [\d.]+-[\d.]+\)/)
    })

    it("does not judge a metric with no better direction", () => {
        const row = renderMarkdownDiff(abRecords(), "batch")
            .split("\n")
            .find((line) => line.startsWith("| final words |"))!

        expect(row).toContain("| - |")
        expect(row).not.toContain("better")
        expect(row).not.toContain("worse")
    })

    it("calls a delta that rounds away to zero the same, not better", () => {
        // 0.4 ms prints as 0, and a rounding artefact labelled "better" is how one becomes a decision
        const records = [makeRecord({ variantId: "batch", startupMs: 1000 }), makeRecord({ variantId: "stream", startupMs: 999.6 })]
        const row = renderMarkdownDiff(records, "batch")
            .split("\n")
            .find((line) => line.startsWith("| startup (ms) |"))!

        expect(row).toContain("| same |")
        expect(row).not.toContain("better")
    })

    it("drops metrics no variant scored", () => {
        expect(renderMarkdownDiff(abRecords(), "batch")).not.toContain("vocabulary WER")
    })

    it("names the variants it does have when the baseline is not one of them", () => {
        expect(() => renderMarkdownDiff(abRecords(), "whisper")).toThrow(/baseline variant "whisper" has no records; present: batch, stream/)
    })
})

describe("bench/report file", () => {
    const outDir = () => fs.mkdtempSync(path.join(os.tmpdir(), "freeshow-bench-report-"))

    it("creates the directory and returns the path written", () => {
        const dir = path.join(outDir(), "nested", "ai-bench")
        const file = writeReport(abRecords(), dir)

        expect(file.startsWith(dir)).toBe(true)
        expect(fs.existsSync(file)).toBe(true)
        expect(path.basename(file)).toBe("smoke-max-2023-11-14T22-13-20Z.json")
    })

    it("writes the records, the environment and the warnings", () => {
        const file = writeReport([makeRecord({ wer: 0.1, resampledFrom: 48000 })], outDir())
        const report = JSON.parse(fs.readFileSync(file, "utf8"))

        expect(report.schema).toBe(1)
        expect(report.generatedAtMs).toBe(1700000000000)
        expect(report.fixtureSets).toEqual([{ id: "smoke", hash: "aaaabbbbcccc" }])
        expect(report.environments).toEqual(["darwin/arm64"])
        expect(report.records).toHaveLength(1)
        expect(report.warnings[0]).toMatch(/^resample:/)
    })

    it("drops the per-word alignment backtrace but keeps its counts", () => {
        const file = writeReport([makeRecord({ wer: 0.1 })], outDir())
        const report = JSON.parse(fs.readFileSync(file, "utf8"))

        expect(report.records[0].metrics.wer.alignment.pairs).toBeUndefined()
        expect(report.records[0].metrics.wer.alignment.refLength).toBe(100)
        expect(report.records[0].metrics.wer.wer).toBe(0.1)
    })

    it("keeps the detection score, including what was missed and what was spurious", () => {
        const file = writeReport([makeRecord({ detection: { expected: 2, found: 1, falsePositives: 1, latencyMs: 1200 } })], outDir())
        const report = JSON.parse(fs.readFileSync(file, "utf8"))

        expect(report.records[0].detection.recall).toBe(0.5)
        expect(report.records[0].detection.matches[0].latencyMs).toBe(1200)
        expect(report.records[0].detection.missed).toHaveLength(1)
        expect(report.records[0].detection.spurious).toHaveLength(1)
    })

    it("refuses to write an empty report", () => {
        expect(() => writeReport([], outDir())).toThrow(/no records/)
        expect(renderConsoleReport([])).toContain("no records")
    })

    it("defaults to the gitignored test-output directory without creating it", () => {
        expect(resolveReportDir().endsWith(path.join("test-output", "ai-bench"))).toBe(true)
    })
})

describe("bench/report aggregation", () => {
    it("pools the lag mean by word count and takes the worst max", () => {
        const records = [makeRecord({ fixtureId: "a", lag: [100] }), makeRecord({ fixtureId: "b", lag: [400, 400, 400] })]
        const summary = summarizeByVariant(records)[0]

        expect(summary.runs).toBe(2)
        expect(summary.fixtures).toBe(2)
        expect(summary.words).toBe(4)
        expect(summary.lagMeanMs).toBe((100 + 400 * 3) / 4) // not (100 + 400) / 2
        expect(summary.lagMaxMs).toBe(400)
    })

    it("counts echoes per minute of audio, not per run", () => {
        const summary = summarizeByVariant([makeRecord({ echoes: 3, audioDurationMs: 30000 })])[0]
        expect(summary.echoesPerMinute.value).toBe(6)
    })

    it("keeps variants in first-appearance order so the baseline stays first", () => {
        const records = [makeRecord({ variantId: "stream" }), makeRecord({ variantId: "batch" })]
        expect(summarizeByVariant(records).map((summary) => summary.variantId)).toEqual(["stream", "batch"])
    })

    it("totals engine errors rather than hiding them in an average", () => {
        const summary = summarizeByVariant([makeRecord({ errors: ["decoder died"] }), makeRecord({ fixtureId: "b" })])[0]
        expect(summary.errors).toBe(1)
    })
})

describe("writeReport file naming", () => {
    const record = (variantId: string): RunRecord => ({
        fixtureSetId: "s",
        fixtureSetHash: "h",
        fixtureId: "f",
        variantId,
        mode: "max",
        platform: "darwin",
        arch: "arm64",
        timestampMs: 1000,
        metrics: { fixtureId: "f", variantId, audioDurationMs: 1000, interimEchoes: [], decodeCostRatio: 0.1, startupMs: 1, errors: [] }
    })

    it("keeps single-variant slices in separate files", () => {
        // bench.sh runs one variant per process to bound memory; every slice of a matrix shares its
        // set, mode and timestamp, so without the variant in the name only the last one survives
        const dir = fs.mkdtempSync(path.join(os.tmpdir(), "freeshow-report-"))
        const a = writeReport([record("stream en-1120")], dir)
        const b = writeReport([record("stream multi-320")], dir)

        expect(a).not.toBe(b)
        expect(fs.readdirSync(dir)).toHaveLength(2)
    })

    it("leaves a multi-variant run's name alone", () => {
        const dir = fs.mkdtempSync(path.join(os.tmpdir(), "freeshow-report-"))
        const file = writeReport([record("a"), record("b")], dir)
        expect(path.basename(file)).toBe("s-max-1970-01-01T00-00-01Z.json")
    })
})

describe("bench/report detection", () => {
    /** The shape of the real set: one sermon that lists references, one clip that lists none. */
    function detectionRecords(): RunRecord[] {
        return [makeRecord({ variantId: "batch", fixtureId: "sermon", wer: 0.1, detection: { expected: 3, found: 3, latencyMs: 900 } }), makeRecord({ variantId: "batch", fixtureId: "worship", wer: 0.2, detection: { expected: 0 } }), makeRecord({ variantId: "stream", fixtureId: "sermon", wer: 0.1, detection: { expected: 3, found: 2, falsePositives: 1, latencyMs: 2400 } }), makeRecord({ variantId: "stream", fixtureId: "worship", wer: 0.2, detection: { expected: 0, falsePositives: 2 } })]
    }

    it("pools recall over references, not over fixtures", () => {
        // three found out of four asked for is 75%; averaging 100% and 0% per fixture would say 50%
        // and let a one-reference clip outvote a three-reference sermon
        const summary = summarizeByVariant([makeRecord({ fixtureId: "a", detection: { expected: 3, found: 3 } }), makeRecord({ fixtureId: "b", detection: { expected: 1, found: 0 } })])[0]

        expect(summary.detection?.expectedReferences).toBe(4)
        expect(summary.detection?.matched).toBe(3)
        expect(summary.detection?.recall?.value).toBe(0.75)
        expect(summary.detection?.recall?.n).toBe(4) // references, not runs
        expect(summary.detection?.referenceRuns).toBe(2)
    })

    it("has no recall at all for a fixture that lists no reference, rather than zero", () => {
        const summary = summarizeByVariant([makeRecord({ audioDurationMs: 30000, detection: { expected: 0, falsePositives: 2 } })])[0]

        expect(summary.detection?.recall).toBeUndefined()
        expect(summary.detection?.expectedReferences).toBe(0)
        expect(summary.detection?.referenceRuns).toBe(0)
        expect(summary.detection?.precision?.value).toBe(0)
        expect(summary.detection?.falsePositivesPerMinute.value).toBe(4)
    })

    it("keeps a reference-free fixture out of recall but inside the false-positive rate", () => {
        const summary = summarizeByVariant([makeRecord({ fixtureId: "a", detection: { expected: 2, found: 2 } }), makeRecord({ fixtureId: "b", detection: { expected: 0, falsePositives: 1 } })])[0]

        expect(summary.detection?.recall?.value).toBe(1)
        expect(summary.detection?.recall?.n).toBe(2)
        expect(summary.detection?.referenceRuns).toBe(1)
        expect(summary.detection?.falsePositivesPerMinute.value).toBe(0.5) // 0/min and 1/min, over a minute each
    })

    it("pools the latency percentiles from every match, not from per-run percentiles", () => {
        const summary = summarizeByVariant([makeRecord({ fixtureId: "a", detection: { expected: 1, found: 1, latencyMs: 400 } }), makeRecord({ fixtureId: "b", detection: { expected: 1, found: 1, latencyMs: 3000 } })])[0]

        expect(summary.detection?.latencyMs.n).toBe(2)
        expect(summary.detection?.latencyMs.p50).toBe(400)
        expect(summary.detection?.latencyMs.max).toBe(3000)
    })

    it("prints recall, precision, false positives and latency in the console table", () => {
        const table = renderConsoleReport([makeRecord({ wer: 0.1, detection: { expected: 2, found: 1, falsePositives: 1, latencyMs: 1200 } })])
        const row = table.split("\n").find((line) => line.trimStart().startsWith("batch "))!

        expect(table).toContain("recall CI")
        expect(table).toContain("det50")
        expect(table).toContain("| 2 expected ref(s) ===")
        expect(row).toContain("50.0%") // one of two found, and one of two detections spurious
        expect(row).toContain("1200")
    })

    it('prints "-" where nothing was asked for, and 0.0% where something was asked for and missed', () => {
        const row = (record: RunRecord) =>
            renderConsoleReport([record])
                .split("\n")
                .find((line) => line.trimStart().startsWith("batch "))!

        // nothing was asked for: a 0% here would read as the feature failing at something
        expect(row(makeRecord({ detection: { expected: 0 } }))).not.toContain("%")
        expect(row(makeRecord({ detection: { expected: 2, found: 0 } }))).toContain("0.0%")
    })

    it("gives the detection columns the room the echo columns had, and only when there is detection", () => {
        expect(renderConsoleReport([makeRecord({ wer: 0.1 })])).toContain("echo CI")
        expect(renderConsoleReport([makeRecord({ wer: 0.1 })])).not.toContain("recall")

        const withDetection = renderConsoleReport([makeRecord({ wer: 0.1, detection: { expected: 1, found: 1 } })])
        expect(withDetection).toContain("recall CI")
        expect(withDetection).not.toContain("echo")
    })

    it("stays inside 120 columns with the detection columns and realistic variant ids", () => {
        const records = ["nemotron-batch-en", "nemotron-stream-en", "nemotron-stream-en-ctc"].flatMap((variantId) => [0, 1].map((index) => makeRecord({ variantId, fixtureId: `fixture-${index}`, wer: 0.12, detection: { expected: 3, found: 2 + index, falsePositives: 1, latencyMs: 1200 + index * 300 } })))

        for (const line of renderConsoleReport(records).split("\n")) expect(line.length).toBeLessThanOrEqual(120)
    })

    it("leads the markdown diff with the product metric and labels its direction", () => {
        const markdown = renderMarkdownDiff(detectionRecords(), "batch")
        const row = (label: string) => markdown.split("\n").find((line) => line.startsWith(`| ${label} |`))!

        expect(markdown.indexOf("| detection recall |")).toBeLessThan(markdown.indexOf("| WER |"))
        expect(row("detection recall")).toContain("higher")
        expect(row("detection recall")).toContain("-33.3 pp **worse**")
        expect(row("detection precision")).toContain("-60.0 pp **worse**") // 2 of 5 detections were asked for
        expect(row("false positives / min")).toContain("+1.50 **worse**")
        expect(row("detection latency p50 (ms)")).toContain("+1500 **worse**")
    })

    it("names how many references recall was pooled over, and over how many runs", () => {
        expect(renderMarkdownDiff(detectionRecords(), "batch")).toContain("recall is over 3 expected reference(s) in 1 of 2 run(s)")
    })

    it("says there is no recall to report when no fixture listed a reference", () => {
        const records = [makeRecord({ variantId: "batch", detection: { expected: 0 } }), makeRecord({ variantId: "stream", detection: { expected: 0, falsePositives: 1 } })]
        const markdown = renderMarkdownDiff(records, "batch")

        expect(markdown).toContain("no recall to report")
        expect(markdown).not.toContain("| detection recall |")
        expect(markdown).toContain("| false positives / min |")
    })

    it("does not report a latency for a variant that matched nothing", () => {
        // describeDistribution of no samples is all zeros, and printing 0 ms would read as instant
        const records = [makeRecord({ variantId: "batch", detection: { expected: 2, found: 0 } }), makeRecord({ variantId: "stream", detection: { expected: 2, found: 1, latencyMs: 800 } })]
        const row = renderMarkdownDiff(records, "batch")
            .split("\n")
            .find((line) => line.startsWith("| detection latency p50 (ms) |"))!

        expect(row).toBe("| detection latency p50 (ms) | lower | - | 800 | - |")
    })
})
