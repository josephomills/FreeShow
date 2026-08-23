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

const mean = (values) => (values.length ? values.reduce((sum, v) => sum + v, 0) / values.length : NaN)

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

    const withRef = covered.filter((key) => records.some((r) => `${r.fixtureSetId}/${r.fixtureId}` === key && r.metrics.wer))
    console.log(`WER pooled over ${withRef.length} fixture(s) that have a reference transcript\n`)

    const rows = variants.map((variant) => {
        const mine = records.filter((r) => r.variantId === variant && covered.includes(`${r.fixtureSetId}/${r.fixtureId}`))
        const decode = mine.map((r) => r.metrics.decodeCostRatio)
        const wer = mine.filter((r) => r.metrics.wer).map((r) => r.metrics.wer.wer)
        const lagMean = mine.map((r) => r.metrics.commitLag.lag.mean)
        const lagMax = mine.map((r) => r.metrics.commitLag.lag.max)
        const echo = mine.map((r) => r.metrics.interimEchoes.length)
        const werCi = ci(wer)

        return [variant, mine.length, mean(decode).toFixed(3) + "x", Math.round(mean(lagMean)), Math.round(Math.max(...lagMax)), wer.length ? (mean(wer) * 100).toFixed(1) + "%" : "-", werCi ? `${(werCi.low * 100).toFixed(1)}-${(werCi.high * 100).toFixed(1)}` : "n<2", echo.reduce((a, b) => a + b, 0)]
    })

    console.log(table(rows, ["variant", "n", "decode", "lag mean", "lag max", "WER", "WER 95% CI", "echo"]))
    console.log(`\ndecode = CPU per second of audio. lag = ms a word is visible as interim before it is committed;`)
    console.log(`detection only ever sees committed text. echo = a word visibly repeated on screen.`)
    console.log(`\nWER here is against a whisper large-v3 PSEUDO-reference on the sermon fixtures. It measures`)
    console.log(`agreement with whisper, not truth, and is only meaningful for ranking these variants`)
    console.log(`against each other. Overlapping CIs mean the difference is not resolved at this n.`)
}

main()
