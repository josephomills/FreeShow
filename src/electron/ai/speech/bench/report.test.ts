// AI BENCH - report guardrails and formatting. Pure functions plus one temp-dir write, so this
// always runs; nothing here needs a model or a fixture.

import fs from "fs"
import os from "os"
import path from "path"
import { describe, expect, it } from "vitest"
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
}

function makeRecord(sample: Sample = {}): RunRecord {
    const audioDurationMs = sample.audioDurationMs ?? 60000
    const lag = sample.lag ?? [100, 200, 300]

    const metrics: RunMetrics = {
        fixtureId: sample.fixtureId ?? "fixture-a",
        variantId: sample.variantId ?? "batch",
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
        fixtureId: sample.fixtureId ?? "fixture-a",
        variantId: sample.variantId ?? "batch",
        mode: sample.mode ?? "max",
        platform: sample.platform ?? "darwin",
        arch: sample.arch ?? "arm64",
        timestampMs: 1700000000000,
        ...(sample.resampledFrom ? { resampledFrom: sample.resampledFrom } : {}),
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
