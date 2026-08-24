// AI BENCH - turn a pile of RunMetrics into a comparison.
//
// Comparison is the whole point of the harness: a single run's numbers decide nothing. These
// fixture sets are tens of clips, the variants are close, and two points of WER between them is
// noise until something says otherwise. Two rules follow, and they are the only reason this file
// exists instead of a console.log at the call site:
//
// 1. Every rate is printed with its n and a bootstrap interval (stats.ts), so a 2-point
//    difference over 6 clips cannot be read as a result.
// 2. Records from different fixture-set hashes, different machines or different pacing modes are
//    refused outright. A table that quietly mixes them is worse than no table, because someone
//    acts on it.
//
// Nothing here reads the clock: the timestamp is passed in by the caller, so the same records
// render byte-identically twice.

import fs from "fs"
import path from "path"
import { summarizeDetection, type DetectionScore } from "./detection"
import type { RunMetrics } from "./metrics"
import type { PaceMode } from "./pacer"
import type { RunResult } from "./runner"
import { bootstrapCi, mean, percentile, type Distribution } from "./stats"

export interface RunRecord {
    fixtureSetId: string
    /** Manifest hash. Two hashes means two sets of reference words - not one comparison. */
    fixtureSetHash: string
    fixtureId: string
    variantId: string
    mode: PaceMode
    platform: string
    arch: string
    /** Injected by the caller. A report must render identically twice. */
    timestampMs: number
    /** Set when the fixture was not natively 16 kHz - a WER caveat, not a blocker. */
    resampledFrom?: number
    /**
     * The product metric for this run - whether the reference the speaker asked for was found, and
     * how late. Replayed from the event log (detection.ts), so a record without it is a
     * transcription-only run rather than a run where detection failed.
     */
    detection?: DetectionScore
    metrics: RunMetrics
}

export function toRunRecord(result: RunResult, metrics: RunMetrics, fixtureSet: { id: string; hash: string }, timestampMs: number, detection?: DetectionScore): RunRecord {
    return {
        fixtureSetId: fixtureSet.id,
        fixtureSetHash: fixtureSet.hash,
        fixtureId: result.fixtureId,
        variantId: result.variantId,
        mode: result.mode,
        platform: result.platform,
        arch: result.arch,
        timestampMs,
        ...(result.resampledFrom ? { resampledFrom: result.resampledFrom } : {}),
        ...(detection ? { detection } : {}),
        metrics
    }
}

// GUARDRAIL 1 - what may be put in the same table

export interface ComparabilityIssue {
    field: "fixtureSet" | "environment" | "mode" | "resample"
    /** Blocking issues make the comparison meaningless. Warnings only make it dirtier. */
    blocking: boolean
    reason: string
}

const environmentOf = (record: RunRecord) => `${record.platform}/${record.arch}`
const distinct = <T>(values: T[]) => [...new Set(values)]

export function checkComparable(records: RunRecord[]): ComparabilityIssue[] {
    const issues: ComparabilityIssue[] = []
    if (!records.length) return issues

    const sets = distinct(records.map((record) => `${record.fixtureSetId}@${record.fixtureSetHash}`))
    if (sets.length > 1) {
        issues.push({
            field: "fixtureSet",
            blocking: true,
            reason: `fixture sets differ (${sets.join(", ")}). WER is scored against the reference transcripts in the manifest, so two manifests are answers to two different questions and the delta between them measures the fixtures.`
        })
    }

    const environments = distinct(records.map(environmentOf))
    if (environments.length > 1) {
        issues.push({
            field: "environment",
            blocking: true,
            reason: `runs came from different machines (${environments.join(", ")}). startup, push-blocked time and drift are properties of the hardware, not of the engine.`
        })
    }

    const modes = distinct(records.map((record) => record.mode))
    if (modes.length > 1) {
        issues.push({
            field: "mode",
            blocking: true,
            reason: `pacing modes differ (${modes.join(", ")}). "rt" feeds audio in real time and "max" as fast as the driver accepts it, so decode cost, wall time and drift are not the same measurement.`
        })
    }

    const resampled = distinct(records.filter((record) => record.resampledFrom).map((record) => `${record.fixtureId}@${record.resampledFrom}Hz`))
    if (resampled.length) {
        issues.push({
            field: "resample",
            blocking: false,
            reason: `resampled to 16 kHz: ${resampled.join(", ")}. Fine for a like-for-like comparison, but the absolute WER is not this engine's number on native 16 kHz audio.`
        })
    }

    return issues
}

export function assertComparable(records: RunRecord[]) {
    const blocking = checkComparable(records).filter((issue) => issue.blocking)
    if (!blocking.length) return

    throw new Error(`refusing to render a comparison across incomparable runs:\n${blocking.map((issue) => `  - ${issue.field}: ${issue.reason}`).join("\n")}`)
}

// GUARDRAIL 2 - no rate without its n and its interval

export interface RateSummary {
    value: number
    /** Runs the interval was resampled from. Small n is the whole reason it is printed. */
    n: number
    low: number
    high: number
}

function summarizeRate(values: number[]): RateSummary {
    const ci = bootstrapCi(values, mean)
    return { value: mean(values), n: values.length, low: ci.low, high: ci.high }
}

/** True when two intervals touch - the delta between the two values is not evidence of anything. */
export function intervalsOverlap(a: RateSummary | undefined, b: RateSummary | undefined): boolean {
    if (!a || !b) return false
    if (![a.low, a.high, b.low, b.high].every(Number.isFinite)) return false
    return a.low <= b.high && b.low <= a.high
}

// SUMMARY

export interface DetectionRates {
    /** Runs whose fixture listed at least one reference. Recall is a statement about these only. */
    referenceRuns: number
    expectedReferences: number
    matched: number
    falsePositives: number
    /**
     * Absent - never 0% - when no fixture in the group listed a reference: silence about a
     * question nobody asked is not a failure to answer it.
     *
     * Its n counts expected references rather than runs, because recall is the mean of one
     * hit-or-miss per reference. That is the weighting a reader wants: a clip with one mention
     * cannot outvote a sermon with six.
     */
    recall?: RateSummary
    /** Absent when nothing at all was detected. n counts the detections that were judged. */
    precision?: RateSummary
    /**
     * The mean of each run's own rate, weighting every fixture equally - as echoesPerMinute does,
     * and the two are read side by side. The reference-free runs are in it deliberately: a clip
     * nobody asked a verse from is the only honest measure of how often the feature interrupts.
     */
    falsePositivesPerMinute: RateSummary
    /** Pooled from every match's own latency, so these percentiles are exact rather than medians. */
    latencyMs: Distribution
}

/**
 * The pooled counts come from summarizeDetection (detection.ts owns what pools how). What is added
 * here is guardrail 2: an interval for each rate, resampled from the same units the rate is a mean
 * of - references for recall, judged detections for precision - so the interval is centred on the
 * headline number instead of on a per-fixture average that would sit somewhere else.
 */
function summarizeDetectionRates(records: RunRecord[]): DetectionRates | undefined {
    const scores = records.map((record) => record.detection).filter((score): score is DetectionScore => !!score)
    if (!scores.length) return undefined

    const pooled = summarizeDetection(scores)
    const hits = scores.flatMap((score) => [...score.matches.map(() => 1), ...score.missed.map(() => 0)])
    const judged = scores.flatMap((score) => [...score.matches.map(() => 1), ...score.spurious.map(() => 0)])

    return {
        referenceRuns: pooled.referenceFixtures,
        expectedReferences: pooled.expectedCount,
        matched: pooled.matched,
        falsePositives: pooled.falsePositives,
        ...(hits.length ? { recall: summarizeRate(hits) } : {}),
        ...(judged.length ? { precision: summarizeRate(judged) } : {}),
        falsePositivesPerMinute: summarizeRate(scores.map((score) => (score.audioDurationMs ? score.falsePositives / (score.audioDurationMs / 60000) : 0))),
        latencyMs: pooled.latencyMs
    }
}

export interface VariantSummary {
    variantId: string
    /** Runs aggregated. Every interval below is drawn from exactly these. */
    runs: number
    fixtures: number
    words: number
    errors: number
    startupMs: number
    lagMeanMs: number
    lagP50Ms: number
    lagP95Ms: number
    lagMaxMs: number
    decodeCost: RateSummary
    echoesPerMinute: RateSummary
    wer?: RateSummary
    vocabulary?: RateSummary
    /** Absent when no run in the group replayed detection. */
    detection?: DetectionRates
}

/** First-appearance order, so the caller's variant ordering (baseline first) survives. */
function groupBy<T>(items: T[], key: (item: T) => string): { key: string; items: T[] }[] {
    const groups: { key: string; items: T[] }[] = []
    for (const item of items) {
        const value = key(item)
        const existing = groups.find((group) => group.key === value)
        if (existing) existing.items.push(item)
        else groups.push({ key: value, items: [item] })
    }
    return groups
}

export function groupByFixture(records: RunRecord[]): { fixtureId: string; records: RunRecord[] }[] {
    return groupBy(records, (record) => record.fixtureId).map((group) => ({ fixtureId: group.key, records: group.items }))
}

export function summarizeVariant(variantId: string, records: RunRecord[]): VariantSummary {
    const lags = records.map((record) => record.metrics.commitLag).filter((lag): lag is NonNullable<typeof lag> => !!lag)

    // a mean IS poolable from summaries (mean_i * n_i sums exactly); percentiles are not, so the
    // aggregate p50/p95 are medians of the per-run percentiles and only exact when runs === 1.
    // max is a max of maxes, which is exact either way.
    const lagSamples = lags.reduce((total, lag) => total + lag.lag.n, 0)
    const lagMeanMs = lagSamples ? lags.reduce((total, lag) => total + lag.lag.mean * lag.lag.n, 0) / lagSamples : 0
    const middleOf = (values: number[]) =>
        percentile(
            [...values].sort((a, b) => a - b),
            50
        )

    const werValues = records.map((record) => record.metrics.wer?.wer).filter((value): value is number => value !== undefined)
    const vocabularyValues = records.map((record) => record.metrics.vocabulary?.rate).filter((value): value is number => value !== undefined)
    const detection = summarizeDetectionRates(records)

    return {
        variantId,
        runs: records.length,
        fixtures: distinct(records.map((record) => record.fixtureId)).length,
        words: lags.reduce((total, lag) => total + lag.words, 0),
        errors: records.reduce((total, record) => total + record.metrics.errors.length, 0),
        startupMs: mean(records.map((record) => record.metrics.startupMs)),
        lagMeanMs,
        lagP50Ms: middleOf(lags.map((lag) => lag.lag.p50)),
        lagP95Ms: middleOf(lags.map((lag) => lag.lag.p95)),
        lagMaxMs: lags.reduce((worst, lag) => Math.max(worst, lag.lag.max), 0),
        decodeCost: summarizeRate(records.map((record) => record.metrics.decodeCostRatio)),
        echoesPerMinute: summarizeRate(records.map((record) => (record.metrics.audioDurationMs ? record.metrics.interimEchoes.length / (record.metrics.audioDurationMs / 60000) : 0))),
        ...(werValues.length ? { wer: summarizeRate(werValues) } : {}),
        ...(vocabularyValues.length ? { vocabulary: summarizeRate(vocabularyValues) } : {}),
        ...(detection ? { detection } : {})
    }
}

export function summarizeByVariant(records: RunRecord[]): VariantSummary[] {
    return groupBy(records, (record) => record.variantId).map((group) => summarizeVariant(group.key, group.items))
}

// FORMATTING

export const formatMs = (value: number) => (Number.isFinite(value) ? `${Math.round(value)}` : "-")
export const formatRatio = (value: number) => (Number.isFinite(value) ? value.toFixed(3) : "-")
export const formatPercent = (value: number) => (Number.isFinite(value) ? `${(value * 100).toFixed(1)}%` : "-")
export const formatRate = (value: number) => (Number.isFinite(value) ? value.toFixed(2) : "-")
/** Percent without the sign, for the two ends of an interval that already carries one. */
const percentValue = (value: number) => (Number.isFinite(value) ? (value * 100).toFixed(1) : "-")

/** An interval, or an explicit "n=1" - never a fabricated range from a single point. */
export function formatCi(rate: RateSummary | undefined, format: (value: number) => string): string {
    if (!rate) return "-"
    if (!Number.isFinite(rate.low) || !Number.isFinite(rate.high)) return `n=${rate.n}`
    return `${format(rate.low)}-${format(rate.high)}`
}

interface Column {
    header: string
    align: "left" | "right"
}

/** Fixed-width table. Widths come from the content, so a long variant id never breaks alignment. */
export function renderTable(columns: Column[], rows: string[][], indent = "  "): string {
    const widths = columns.map((column, index) => Math.max(column.header.length, ...rows.map((row) => (row[index] ?? "").length)))
    const line = (cells: string[]) => (indent + cells.map((cell, index) => (columns[index].align === "left" ? cell.padEnd(widths[index]) : cell.padStart(widths[index]))).join("  ")).trimEnd()

    return [line(columns.map((column) => column.header)), indent + widths.map((width) => "-".repeat(width)).join("  "), ...rows.map(line)].join("\n")
}

// REPORT FILE

export const REPORT_SCHEMA = 1

export interface BenchReport {
    schema: number
    /** From the records, never Date.now(). */
    generatedAtMs: number
    fixtureSets: { id: string; hash: string }[]
    environments: string[]
    modes: PaceMode[]
    /** Every issue found, blocking ones marked - so a written report can never look clean. */
    warnings: string[]
    records: RunRecord[]
}

export function buildReport(records: RunRecord[]): BenchReport {
    if (!records.length) throw new Error("nothing to report: no records")

    return {
        schema: REPORT_SCHEMA,
        generatedAtMs: Math.max(...records.map((record) => record.timestampMs)),
        fixtureSets: groupBy(records, (record) => `${record.fixtureSetId}@${record.fixtureSetHash}`).map((group) => ({ id: group.items[0].fixtureSetId, hash: group.items[0].fixtureSetHash })),
        environments: distinct(records.map(environmentOf)),
        modes: distinct(records.map((record) => record.mode)),
        warnings: checkComparable(records).map((issue) => `${issue.blocking ? "BLOCKING " : ""}${issue.field}: ${issue.reason}`),
        records
    }
}

/** test-output/ is gitignored, and the bench is run from the repo root. */
export function resolveReportDir(): string {
    return path.join(process.cwd(), "test-output", "ai-bench")
}

const fileSafe = (value: string) => value.replace(/[^a-zA-Z0-9._-]+/g, "-")

/**
 * The alignment backtrace is one entry per reference word. Priceless while analysing one run,
 * megabytes of noise in a comparison file - the counts it was derived from stay.
 */
const withoutAlignmentPairs = (key: string, value: unknown) => (key === "pairs" ? undefined : value)

export function writeReport(records: RunRecord[], outDir = resolveReportDir()): string {
    const report = buildReport(records)
    fs.mkdirSync(outDir, { recursive: true })

    const stamp = new Date(report.generatedAtMs)
        .toISOString()
        .replace(/\.\d+Z$/, "Z")
        .replace(/:/g, "-")
    // The variant is part of the name because bench.sh runs one variant per process to bound
    // memory, and every slice of a matrix shares its set, mode and timestamp - without this they
    // all write the same path and only the last variant survives. Omitted when a run covers
    // several variants, so a whole-matrix run keeps its original name.
    const variants = [...new Set(records.map((record) => record.variantId))]
    const variantPart = variants.length === 1 ? `-${fileSafe(variants[0])}` : ""
    const file = path.join(outDir, `${fileSafe(report.fixtureSets.map((set) => set.id).join("+"))}-${fileSafe(report.modes.join("+"))}${variantPart}-${stamp}.json`)
    fs.writeFileSync(file, JSON.stringify(report, withoutAlignmentPairs, 2))

    return file
}

// CONSOLE

/**
 * A terminal has a fixed budget, so the detection block is not free: when a run scored detection -
 * the product metric - it takes the room that the lag tail and the interim echoes had. Both are
 * still in the markdown diff and in the JSON, and the choice is deliberate: a reader comparing two
 * variants wants to know whether the verse was found before they want the echo rate.
 */
function variantRows(summaries: VariantSummary[]): { columns: Column[]; rows: string[][] } {
    const withVocabulary = summaries.some((summary) => summary.vocabulary)
    const withDetection = summaries.some((summary) => summary.detection)
    const latency = (summary: VariantSummary, pick: (distribution: Distribution) => number) => (summary.detection?.latencyMs.n ? formatMs(pick(summary.detection.latencyMs)) : "-")

    const columns: Column[] = [
        { header: "variant", align: "left" },
        { header: "n", align: "right" },
        { header: "decode", align: "right" },
        { header: "lag mean", align: "right" },
        ...(withDetection ? [] : ([{ header: "lag p95", align: "right" }] as Column[])),
        { header: "WER", align: "right" },
        { header: "WER 95% CI", align: "right" },
        ...(withVocabulary
            ? ([
                  { header: "vocab", align: "right" },
                  { header: "vocab CI", align: "right" }
              ] as Column[])
            : []),
        ...(withDetection
            ? ([
                  { header: "recall", align: "right" },
                  { header: "recall CI", align: "right" },
                  { header: "prec", align: "right" },
                  { header: "fp/min", align: "right" },
                  { header: "det50", align: "right" },
                  { header: "det95", align: "right" }
              ] as Column[])
            : ([
                  { header: "echo/min", align: "right" },
                  { header: "echo CI", align: "right" }
              ] as Column[])),
        { header: "err", align: "right" }
    ]

    const rows = summaries.map((summary) => [
        summary.variantId,
        `${summary.runs}`,
        `${formatRatio(summary.decodeCost.value)}x`,
        formatMs(summary.lagMeanMs),
        ...(withDetection ? [] : [formatMs(summary.lagP95Ms)]),
        summary.wer ? formatPercent(summary.wer.value) : "-",
        formatCi(summary.wer, percentValue),
        ...(withVocabulary ? [summary.vocabulary ? formatPercent(summary.vocabulary.value) : "-", formatCi(summary.vocabulary, percentValue)] : []),
        ...(withDetection
            ? [
                  // "-" and not "0.0%": this fixture never asked for a verse, so there was nothing to recall
                  summary.detection?.recall ? formatPercent(summary.detection.recall.value) : "-",
                  formatCi(summary.detection?.recall, percentValue),
                  summary.detection?.precision ? formatPercent(summary.detection.precision.value) : "-",
                  summary.detection ? formatRate(summary.detection.falsePositivesPerMinute.value) : "-",
                  latency(summary, (distribution) => distribution.p50),
                  latency(summary, (distribution) => distribution.p95)
              ]
            : [formatRate(summary.echoesPerMinute.value), formatCi(summary.echoesPerMinute, formatRate)]),
        `${summary.errors}`
    ])

    return { columns, rows }
}

export function renderVariantTable(summaries: VariantSummary[]): string {
    const { columns, rows } = variantRows(summaries)
    return renderTable(columns, rows)
}

/**
 * One table per fixture, one row per variant, plus a pooled table when there is more than one
 * fixture - that pooled table is the only place an interval has an n worth having.
 */
export function renderConsoleReport(records: RunRecord[]): string {
    if (!records.length) return "[bench] nothing to report: no records"
    assertComparable(records)

    const report = buildReport(records)
    const set = report.fixtureSets[0]
    const groups = groupByFixture(records)
    const variants = distinct(records.map((record) => record.variantId))

    const lines = [`AI BENCH  ${set.id} [${set.hash}]`, `  ${report.environments[0]} | mode=${report.modes[0]} | ${groups.length} fixture(s) | ${variants.length} variant(s) | ${new Date(report.generatedAtMs).toISOString()}`, `  n = runs pooled; 95% CI = bootstrap over those runs. n=1 prints as n=1, not as an interval.`]
    if (records.some((record) => record.detection)) {
        lines.push(`  recall/prec = scripture references found; recall's own n is references, not runs, and`)
        lines.push(`  prints "-" where the fixture listed none. det50/det95 = ms from the spoken phrase`)
        lines.push(`  ending to the reference being actionable. Precision's interval is in the markdown diff.`)
    }
    for (const warning of report.warnings) lines.push(`  ! ${warning}`)

    for (const group of groups) {
        const seconds = (Math.max(...group.records.map((record) => record.metrics.audioDurationMs)) / 1000).toFixed(1)
        const refWords = Math.max(0, ...group.records.map((record) => record.metrics.wer?.refLength ?? 0))
        // named next to the fixture rather than in a column, because it is the reason a recall cell
        // is "-" and a reader should not have to infer that from an empty number
        const expected = Math.max(0, ...group.records.map((record) => record.detection?.expectedCount ?? 0))

        lines.push("")
        lines.push(`=== ${group.fixtureId} | ${seconds}s${refWords ? ` | ${refWords} ref words` : ""}${expected ? ` | ${expected} expected ref(s)` : ""} ===`)
        lines.push(renderVariantTable(summarizeByVariant(group.records)))
    }

    if (groups.length > 1) {
        lines.push("")
        lines.push(`=== all fixtures | n=${groups.length} ===`)
        lines.push(renderVariantTable(summarizeByVariant(records)))
    }

    return lines.join("\n")
}

// MARKDOWN A/B

type Direction = "lower" | "higher" | "none"

interface MetricRow {
    label: string
    direction: Direction
    rate?: (summary: VariantSummary) => RateSummary | undefined
    value?: (summary: VariantSummary) => number | undefined
    format: (value: number) => string
    /** Interval ends, when printing them the same way as the value would repeat the unit. */
    formatBound?: (value: number) => string
    /** Only when the difference is not in the value's own unit - a percent delta is in points. */
    formatDelta?: (delta: number) => string
}

const signed = (delta: number, format: (value: number) => string) => `${delta >= 0 ? "+" : "-"}${format(Math.abs(delta))}`

const pointsDelta = (delta: number) => `${signed(delta, percentValue)} pp`

// Detection leads: WER says how many words the engine got right, these say whether the feature
// worked. A p50 or p95 of "-" means nothing was matched, which is not a latency of zero.
const METRIC_ROWS: MetricRow[] = [
    { label: "detection recall", direction: "higher", rate: (summary) => summary.detection?.recall, format: formatPercent, formatBound: percentValue, formatDelta: pointsDelta },
    { label: "detection precision", direction: "higher", rate: (summary) => summary.detection?.precision, format: formatPercent, formatBound: percentValue, formatDelta: pointsDelta },
    { label: "false positives / min", direction: "lower", rate: (summary) => summary.detection?.falsePositivesPerMinute, format: formatRate },
    { label: "detection latency p50 (ms)", direction: "lower", value: (summary) => (summary.detection?.latencyMs.n ? summary.detection.latencyMs.p50 : undefined), format: formatMs },
    { label: "detection latency p95 (ms)", direction: "lower", value: (summary) => (summary.detection?.latencyMs.n ? summary.detection.latencyMs.p95 : undefined), format: formatMs },
    { label: "WER", direction: "lower", rate: (summary) => summary.wer, format: formatPercent, formatBound: percentValue, formatDelta: pointsDelta },
    { label: "vocabulary WER", direction: "lower", rate: (summary) => summary.vocabulary, format: formatPercent, formatBound: percentValue, formatDelta: pointsDelta },
    { label: "commit lag mean (ms)", direction: "lower", value: (summary) => summary.lagMeanMs, format: formatMs },
    { label: "commit lag p50 (ms)", direction: "lower", value: (summary) => summary.lagP50Ms, format: formatMs },
    { label: "commit lag p95 (ms)", direction: "lower", value: (summary) => summary.lagP95Ms, format: formatMs },
    { label: "commit lag max (ms)", direction: "lower", value: (summary) => summary.lagMaxMs, format: formatMs },
    { label: "decode cost (x audio)", direction: "lower", rate: (summary) => summary.decodeCost, format: formatRatio },
    { label: "interim echoes / min", direction: "lower", rate: (summary) => summary.echoesPerMinute, format: formatRate },
    { label: "startup (ms)", direction: "lower", value: (summary) => summary.startupMs, format: formatMs },
    { label: "engine errors", direction: "lower", value: (summary) => summary.errors, format: (value) => `${value}` },
    { label: "final words", direction: "none", value: (summary) => summary.words, format: (value) => `${value}` }
]

const BETTER_IS: Record<Direction, string> = { lower: "lower", higher: "higher", none: "-" }

function metricValue(row: MetricRow, summary: VariantSummary): number | undefined {
    return row.rate ? row.rate(summary)?.value : row.value?.(summary)
}

function metricCell(row: MetricRow, summary: VariantSummary): string {
    const value = metricValue(row, summary)
    if (value === undefined) return "-"

    const rate = row.rate?.(summary)
    if (!rate) return row.format(value)

    return `${row.format(value)} (n=${rate.n}, 95% CI ${formatCi(rate, row.formatBound ?? row.format)})`
}

/** The sign plus the word, because a reader should never have to remember which way is good. */
function deltaCell(row: MetricRow, baseline: VariantSummary, other: VariantSummary): string {
    const from = metricValue(row, baseline)
    const to = metricValue(row, other)
    if (from === undefined || to === undefined) return "-"

    const delta = to - from
    const format = (value: number) => (row.formatDelta ? row.formatDelta(value) : signed(value, row.format))
    const text = format(delta)

    // "same" is judged on the PRINTED delta, not the raw one: a 0.4 ms difference prints as 0,
    // and calling a number that renders as zero "better" is how a rounding artefact becomes a
    // decision
    const unsigned = (value: string) => value.replace(/^[+-]/, "")
    if (unsigned(text) === unsigned(format(0))) return "same"

    if (row.direction === "none") return text

    const better = row.direction === "lower" ? delta < 0 : delta > 0
    const noise = intervalsOverlap(row.rate?.(baseline), row.rate?.(other)) ? ", within noise" : ""
    return `${text} **${better ? "better" : "worse"}**${noise}`
}

/**
 * Baseline against everything else, one column pair per challenger. Deltas only - the absolute
 * numbers are in the console report and in the JSON.
 */
export function renderMarkdownDiff(records: RunRecord[], baselineVariantId: string): string {
    if (!records.length) throw new Error("nothing to compare: no records")
    assertComparable(records)

    const summaries = summarizeByVariant(records)
    const baseline = summaries.find((summary) => summary.variantId === baselineVariantId)
    if (!baseline) throw new Error(`baseline variant "${baselineVariantId}" has no records; present: ${summaries.map((summary) => summary.variantId).join(", ")}`)

    const challengers = summaries.filter((summary) => summary.variantId !== baselineVariantId)
    const report = buildReport(records)
    const set = report.fixtureSets[0]

    const headers = ["metric", "better when", `\`${baseline.variantId}\` (baseline)`, ...challengers.flatMap((challenger) => [`\`${challenger.variantId}\``, "delta"])]

    const rows = METRIC_ROWS.filter((row) => summaries.some((summary) => metricValue(row, summary) !== undefined)).map((row) => [row.label, BETTER_IS[row.direction], metricCell(row, baseline), ...challengers.flatMap((challenger) => [metricCell(row, challenger), deltaCell(row, baseline, challenger)])])

    const lines = [
        `# AI bench A/B - baseline \`${baseline.variantId}\``,
        "",
        `**${set.id}** \`${set.hash}\` | ${report.environments[0]} | mode \`${report.modes[0]}\` | ${groupByFixture(records).length} fixture(s) | ${records.length} run(s) | ${new Date(report.generatedAtMs).toISOString()}`,
        "",
        `Runs per variant: ${summaries.map((summary) => `\`${summary.variantId}\` n=${summary.runs}`).join(", ")}.`,
        "",
        '> "better when" says which direction wins, so the sign in the delta column reads on its own.',
        "> Rates carry n and a 95% bootstrap interval over those runs; these fixture sets are small,",
        "> and a delta whose intervals overlap is marked *within noise* because it is not a result.",
        ""
    ]

    // Recall's denominator, spelled out: without it a reader cannot tell 100% over three references
    // in one sermon from 100% over a whole fixture set, and the rest of the audio still counts
    // against false positives per minute even though it contributes nothing to recall.
    const detection = baseline.detection
    if (detection) {
        lines.push(detection.expectedReferences ? `> Detection recall is over ${detection.expectedReferences} expected reference(s) in ${detection.referenceRuns} of ${baseline.runs} run(s). The runs that list none score no recall at all - they contribute only false positives per minute.` : `> No fixture in this set lists an expected reference, so there is no recall to report; only false positives per minute is meaningful here.`, "")
    }

    for (const warning of report.warnings) lines.push(`> **Warning** ${warning}`, "")

    lines.push(`| ${headers.join(" | ")} |`)
    lines.push(`| ${headers.map(() => "---").join(" | ")} |`)
    for (const row of rows) lines.push(`| ${row.join(" | ")} |`)

    return lines.join("\n")
}
