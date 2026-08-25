#!/usr/bin/env node
// AI BENCH - pool the per-slice reports into one comparison.
//
// bench.sh runs each (fixture set, variant) in its own process, so a matrix arrives as dozens of
// JSON files. This merges them.
//
// It deliberately does what report.ts's guardrail refuses to: pool across fixture sets. That
// guardrail is right for comparing RUNS - two runs over different audio are not comparable. But
// comparing VARIANTS is the opposite case: every variant saw exactly the same fixtures, so pooling
// across sets is the only way to get an n large enough to say anything. Any variant missing a
// fixture the others have is dropped from the pooled view and reported, because an incomplete row
// would flatter or punish it for free.
//
// Recall gets the same treatment as WER: it is pooled only over the fixtures whose manifest lists
// a reference, and the count of those fixtures is printed. Most clips list none - they still count
// toward false positives per minute, which is the only detection number they can honestly produce.
//
//   node scripts/ai/report.js [--dir test-output/ai-bench] [--stamp <ms>]

const fs = require("fs")
const path = require("path")

function parseArgs(argv) {
    const args = { dir: path.join("test-output", "ai-bench"), stamp: null }
    for (let i = 0; i < argv.length; i++) {
        if (argv[i] === "--dir") args.dir = argv[++i]
        else if (argv[i] === "--stamp") args.stamp = Number(argv[++i])
    }
    return args
}

/** Below this a 120s excerpt is worship, applause or silence rather than preaching. */
const MIN_WPM = 60

const mean = (values) => (values.length ? values.reduce((sum, v) => sum + v, 0) / values.length : NaN)

/** Same rule as stats.ts, so a pooled percentile here and a per-run one there mean the same thing. */
function percentile(values, p) {
    if (!values.length) return NaN
    const sorted = [...values].sort((a, b) => a - b)
    return sorted[Math.min(sorted.length - 1, Math.max(0, Math.ceil((p / 100) * sorted.length) - 1))]
}

/** Deterministic bootstrap CI - the same inputs must render the same interval every time. */
function ci(values, seed = 0x5eed) {
    if (values.length < 2) return null
    let state = seed || 1
    const next = (limit) => {
        state ^= state << 13
        state ^= state >>> 17
        state ^= state << 5
        return Math.abs(state) % limit
    }
    const estimates = []
    for (let i = 0; i < 2000; i++) {
        const sample = Array.from({ length: values.length }, () => values[next(values.length)])
        estimates.push(mean(sample))
    }
    estimates.sort((a, b) => a - b)
    const at = (q) => estimates[Math.min(estimates.length - 1, Math.ceil(q * estimates.length) - 1)]
    return { low: at(0.025), high: at(0.975) }
}

function load(dir, stamp) {
    const records = []
    for (const file of fs.readdirSync(dir).filter((f) => f.endsWith(".json"))) {
        const report = JSON.parse(fs.readFileSync(path.join(dir, file), "utf8"))
        for (const record of report.records) {
            if (stamp && record.timestampMs !== stamp) continue
            records.push(record)
        }
    }
    return records
}

function table(rows, headers) {
    const widths = headers.map((h, i) => Math.max(h.length, ...rows.map((r) => String(r[i]).length)))
    const line = (cells) => "  " + cells.map((c, i) => (i === 0 ? String(c).padEnd(widths[i]) : String(c).padStart(widths[i]))).join("  ")
    return [line(headers), line(widths.map((w) => "-".repeat(w))), ...rows.map(line)].join("\n")
}

function main() {
    const args = parseArgs(process.argv.slice(2))
    if (!fs.existsSync(args.dir)) {
        console.error(`no reports in ${args.dir} - run scripts/ai/bench.sh first`)
        process.exit(1)
    }

    let records = load(args.dir, args.stamp)
    if (!records.length) {
        console.error("no records matched")
        process.exit(1)
    }

    // newest matrix only, unless a stamp was named
    if (!args.stamp) {
        const latest = Math.max(...records.map((r) => r.timestampMs))
        records = records.filter((r) => r.timestampMs === latest)
    }

    const variants = [...new Set(records.map((r) => r.variantId))]
    const fixtures = [...new Set(records.map((r) => `${r.fixtureSetId}/${r.fixtureId}`))]

    // only fixtures every variant covered - a partial row is not a comparison
    const covered = fixtures.filter((key) => variants.every((v) => records.some((r) => `${r.fixtureSetId}/${r.fixtureId}` === key && r.variantId === v)))
    const dropped = fixtures.filter((f) => !covered.includes(f))

    console.log(`\npooled over ${covered.length} fixture(s) x ${variants.length} variant(s), ${records[0].platform}/${records[0].arch}, mode=${records[0].mode}`)
    if (dropped.length) console.log(`dropped ${dropped.length} fixture(s) not run by every variant: ${dropped.slice(0, 4).join(", ")}${dropped.length > 4 ? " ..." : ""}`)

    // Speech-density gate. makeFixtures.js cuts at a fixed offset, which sometimes lands on
    // worship rather than preaching - one 120s excerpt's reference is "Oh, oh, oh" forty-four
    // times. Scoring WER on that measures nothing about transcription and drags the pooled number
    // around, so it is excluded and named rather than silently averaged in. Preaching runs
    // 120-180 wpm; the threshold is well below anything that is actually speech.
    const wpm = (record) => (record.metrics.wer ? record.metrics.wer.refLength / (record.metrics.audioDurationMs / 60000) : NaN)
    const sparse = covered.filter((key) => {
        const any = records.find((r) => `${r.fixtureSetId}/${r.fixtureId}` === key && r.metrics.wer)
        return any && wpm(any) < MIN_WPM
    })
    const scored = covered.filter((key) => !sparse.includes(key) && records.some((r) => `${r.fixtureSetId}/${r.fixtureId}` === key && r.metrics.wer))

    console.log(`WER pooled over ${scored.length} fixture(s) with a reference transcript`)
    if (sparse.length) console.log(`excluded from WER, under ${MIN_WPM} words/min so probably not speech: ${sparse.join(", ")}`)

    // Recall's denominator. A fixture whose manifest lists nothing is not a fixture the feature
    // failed on, so it is kept out of recall entirely - but its audio still counts against false
    // positives per minute, which is the only detection number reference-free audio can produce.
    const withDetection = records.some((r) => r.detection)
    const referenced = covered.filter((key) => records.some((r) => `${r.fixtureSetId}/${r.fixtureId}` === key && r.detection && r.detection.expectedCount > 0))
    if (withDetection) {
        const named = `${referenced.slice(0, 4).join(", ")}${referenced.length > 4 ? " ..." : ""}`
        console.log(referenced.length ? `detection recall pooled over ${referenced.length} of ${covered.length} fixture(s) that list a reference: ${named}` : `no fixture lists an expected reference - recall prints "-" and only fp/min says anything`)
    }
    console.log()

    const rows = variants.map((variant) => {
        const mine = records.filter((r) => r.variantId === variant && covered.includes(`${r.fixtureSetId}/${r.fixtureId}`))
        const decode = mine.map((r) => r.metrics.decodeCostRatio)
        const wall = mine.map((r) => r.metrics.wallCostRatio).filter((v) => Number.isFinite(v))
        const wer = mine.filter((r) => r.metrics.wer && scored.includes(`${r.fixtureSetId}/${r.fixtureId}`)).map((r) => r.metrics.wer.wer)
        const lagMean = mine.map((r) => r.metrics.commitLag.lag.mean)
        const lagMax = mine.map((r) => r.metrics.commitLag.lag.max)
        const echo = mine.map((r) => r.metrics.interimEchoes.length)
        const looped = mine.map((r) => (r.metrics.repetition ? r.metrics.repetition.share : NaN)).filter((v) => Number.isFinite(v))
        const werCi = ci(wer)

        const head = [variant, mine.length, mean(decode).toFixed(3) + "x", wall.length ? mean(wall).toFixed(3) + "x" : "-", looped.length ? (mean(looped) * 100).toFixed(1) + "%" : "-", Math.round(mean(lagMean))]
        const middle = [wer.length ? (mean(wer) * 100).toFixed(1) + "%" : "-", werCi ? `${(werCi.low * 100).toFixed(1)}-${(werCi.high * 100).toFixed(1)}` : "n<2"]
        if (!withDetection) return [...head, Math.round(Math.max(...lagMax)), ...middle, echo.reduce((a, b) => a + b, 0)]

        const detections = mine.map((r) => r.detection).filter(Boolean)
        const expected = detections.reduce((total, d) => total + d.expectedCount, 0)
        const matched = detections.reduce((total, d) => total + d.matched, 0)
        const falsePositives = detections.reduce((total, d) => total + d.falsePositives, 0)
        const judged = matched + falsePositives
        // one sample per expected reference, so the interval is drawn from the same weighting the
        // pooled recall uses - a fixture with six mentions counts for six, not for one
        const recallCi = ci(Array.from({ length: expected }, (_, i) => (i < matched ? 1 : 0)))
        // exact percentiles: every match carries its own latency into the JSON, so nothing is
        // being re-derived from per-run percentiles that could not be pooled
        const latencies = detections.flatMap((d) => d.matches.map((m) => m.latencyMs))
        const fpPerMinute = mean(detections.map((d) => (d.audioDurationMs ? d.falsePositives / (d.audioDurationMs / 60000) : 0)))
        const ms = (value) => (Number.isFinite(value) ? Math.round(value) : "-")

        return [...head, ...middle, expected ? ((matched / expected) * 100).toFixed(1) + "%" : "-", expected ? (recallCi ? `${(recallCi.low * 100).toFixed(1)}-${(recallCi.high * 100).toFixed(1)}` : "n<2") : "-", judged ? ((matched / judged) * 100).toFixed(1) + "%" : "-", fpPerMinute.toFixed(2), ms(percentile(latencies, 50)), ms(percentile(latencies, 95))]
    })

    const headers = withDetection
        ? ["variant", "n", "cpu", "wall", "looped", "lag mean", "WER", "WER 95% CI", "recall", "recall CI", "prec", "fp/min", "det50", "det95"]
        : ["variant", "n", "cpu", "wall", "looped", "lag mean", "lag max", "WER", "WER 95% CI", "echo"]

    console.log(table(rows, headers))
    console.log(`\ncpu  = CPU seconds per second of audio - the engine's real cost. Above 1.0 it cannot keep up.`)
    console.log(`wall = the same thing in wall time. It runs LOWER than cpu because the decoder uses two`)
    console.log(`       threads, so cpu sums across them - cpu is total work, wall is elapsed time.`)
    console.log(`looped = share of the transcript lost to decoder cycles. Only meaningful over LONG audio -`)
    console.log(`       the failure needs minutes of continuous decoding, so short clips always read 0%.`)
    console.log(`lag  = ms a word is visible as interim before it is committed; detection only sees committed`)
    console.log(`       text.`)
    if (!withDetection) {
        console.log(`echo = a word visibly repeated on screen.`)
    } else {
        console.log(`\nrecall = spoken references the feature found, pooled over references rather than over`)
        console.log(`       fixtures, so its n is references. "-" means the fixture asked for none - not 0%.`)
        console.log(`prec / fp/min = of the references it offered, how many were asked for, and how often it`)
        console.log(`       interrupts per minute of audio. fp/min counts EVERY fixture, reference-free ones`)
        console.log(`       included - those clips are the only honest measure of unprompted interruption.`)
        console.log(`det50/det95 = ms from the spoken phrase ending to the reference being actionable.`)
    }
    console.log(`\nWER here is against a whisper large-v3 PSEUDO-reference on the sermon fixtures. It measures`)
    console.log(`agreement with whisper, not truth, and is only meaningful for ranking these variants`)
    console.log(`against each other. Overlapping CIs mean the difference is not resolved at this n.`)
}

main()
