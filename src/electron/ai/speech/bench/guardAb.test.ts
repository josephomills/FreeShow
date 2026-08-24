// AI BENCH - what the hold-until-settled guard costs and what it buys.
//
// The guard stops a reference being projected while it is still being spoken ("matthew 6" is a
// complete reference to Matthew 6:1 right up until "33" arrives). Its cost is latency: the verse
// appears once the speaker says anything else, or once the utterance closes.
//
// Both sides are scored from the SAME decode - the engine runs once per fixture and detection is
// replayed twice - so nothing in the comparison can be attributed to the transcription differing.
//
//   AI_BENCH=1 npx vitest run --config config/testing/vitest.config.ts \
//     src/electron/ai/speech/bench/guardAb.test.ts

import path from "path"
import { describe, expect, it } from "vitest"
import { replayDetection, scoreDetection } from "./detection"
import { availableVariants, benchEngineReady, resolveNemotronModelDir } from "./engines"
import { availableFixtures, listManifests, loadFixtureSet, type Fixture } from "./fixtures"
import { runFixture } from "./runner"
import { bootstrapCi, describe as describeDistribution, mean } from "./stats"

const VARIANT_ID = process.env.AI_BENCH_VARIANT || "stream multi-1120"

const withReferences: Fixture[] = listManifests()
    .flatMap((manifest) => availableFixtures(loadFixtureSet(manifest)))
    .filter((fixture) => fixture.expected.length > 0)

const variant = availableVariants().find((entry) => entry.id === VARIANT_ID)
const canRun = !!process.env.AI_BENCH && benchEngineReady(variant?.modelSet) && !!variant && withReferences.length > 0
const describeIfEngine = canRun ? describe : describe.skip

if (process.env.AI_BENCH && !canRun) {
    console.warn(`[guard a/b] skipped: variant=${VARIANT_ID} model=${resolveNemotronModelDir()} fixtures=${withReferences.length}`)
}

interface Side {
    matched: number
    references: number
    judged: number
    truePositives: number
    premature: number
    latencies: number[]
    /** One 1/0 per expected reference, so the interval is over the unit the rate averages. */
    recallUnits: number[]
    precisionUnits: number[]
}

const empty = (): Side => ({ matched: 0, references: 0, judged: 0, truePositives: 0, premature: 0, latencies: [], recallUnits: [], precisionUnits: [] })

function accumulate(side: Side, score: ReturnType<typeof scoreDetection>) {
    side.matched += score.matches.length
    side.references += score.matches.length + score.missed.length
    side.premature += score.spurious.length
    side.truePositives += score.matches.length
    side.judged += score.matches.length + score.spurious.length
    score.matches.forEach((match) => side.latencies.push(match.latencyMs))
    score.matches.forEach(() => side.recallUnits.push(1))
    score.missed.forEach(() => side.recallUnits.push(0))
    score.matches.forEach(() => side.precisionUnits.push(1))
    score.spurious.forEach(() => side.precisionUnits.push(0))
}

function row(label: string, side: Side): string {
    const latency = describeDistribution(side.latencies)
    const recall = side.references ? side.matched / side.references : NaN
    const precision = side.judged ? side.truePositives / side.judged : NaN
    const rci = bootstrapCi(side.recallUnits, mean)
    const pci = bootstrapCi(side.precisionUnits, mean)
    const pct = (value: number) => (Number.isFinite(value) ? `${(value * 100).toFixed(1)}%` : "-")
    const ci = (interval: { low: number; high: number }) => (Number.isFinite(interval.low) ? `${(interval.low * 100).toFixed(0)}-${(interval.high * 100).toFixed(0)}` : "n<2")

    return `  ${label.padEnd(14)}${pct(recall).padStart(8)}${ci(rci).padStart(10)}${pct(precision).padStart(11)}${ci(pci).padStart(10)}${String(side.premature).padStart(11)}${`${Math.round(latency.mean)}`.padStart(10)}${`${latency.p50}`.padStart(8)}${`${latency.p95}`.padStart(8)}${`${latency.max}`.padStart(8)}`
}

describeIfEngine(`hold-until-settled guard (${VARIANT_ID})`, () => {
    it(`scores ${withReferences.length} fixture(s) with and without the guard`, async () => {
        const held = empty()
        const unheld = empty()

        for (const fixture of withReferences) {
            const result = await runFixture({ fixtureId: fixture.id, fixturePath: fixture.absolutePath, variant: variant!, mode: "max" })

            // one decode, two replays - replayDetection must never run concurrently with itself
            accumulate(held, scoreDetection(await replayDetection(result), fixture.expected))
            accumulate(unheld, scoreDetection(await replayDetection(result, { holdProvisionalReferences: false }), fixture.expected))
        }

        console.log(`\n  ${withReferences.length} fixtures, ${held.references} spoken reference(s), variant ${VARIANT_ID}`)
        console.log(`  ${"".padEnd(14)}${"recall".padStart(8)}${"CI".padStart(10)}${"precision".padStart(11)}${"CI".padStart(10)}${"premature".padStart(11)}${"lat mean".padStart(10)}${"p50".padStart(8)}${"p95".padStart(8)}${"max".padStart(8)}`)
        console.log(`  ${"".padEnd(14)}${"-".repeat(8).padStart(8)}${"-".repeat(10).padStart(10)}${"-".repeat(11).padStart(11)}${"-".repeat(10).padStart(10)}${"-".repeat(11).padStart(11)}${"-".repeat(10).padStart(10)}${"-".repeat(8).padStart(8)}${"-".repeat(8).padStart(8)}${"-".repeat(8).padStart(8)}`)
        console.log(row("without guard", unheld))
        console.log(row("with guard", held))
        console.log(`\n  premature = a wrong passage projected before the right one. latency in ms from the`)
        console.log(`  last word of the spoken phrase to the reference becoming actionable.`)

        // the guard exists to remove premature projections; if it ever stops doing that, say so
        expect(held.premature).toBeLessThanOrEqual(unheld.premature)
    }, 3_600_000)
})
