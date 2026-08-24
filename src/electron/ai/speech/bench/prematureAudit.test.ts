// AI BENCH - what the wrong projections actually are.
//
// Precision says how many projections were wrong; it does not say WHY, and the why decides what can
// be done about it. A reference built from a mis-heard word cannot be fixed by matching more
// carefully - only by transcribing better, or by declining to project on weak evidence. One built
// from a half-spoken reference can.
//
//   AI_BENCH=1 npx vitest run --config config/testing/vitest.config.ts \
//     src/electron/ai/speech/bench/prematureAudit.test.ts

import { describe, expect, it } from "vitest"
import { replayDetection, scoreDetection } from "./detection"
import { availableVariants, benchModelReady, hasSherpa } from "./engines"
import { availableFixtures, listManifests, loadFixtureSet, type Fixture } from "./fixtures"
import { runFixture } from "./runner"

const VARIANT_ID = process.env.AI_BENCH_VARIANT || "multi-1120 warm"

const withReferences: Fixture[] = listManifests()
    .flatMap((manifest) => availableFixtures(loadFixtureSet(manifest)))
    .filter((fixture) => fixture.expected.length > 0)

const variant = availableVariants().find((entry) => entry.id === VARIANT_ID)
const canRun = !!process.env.AI_BENCH && hasSherpa() && !!variant && benchModelReady(variant.modelSet) && withReferences.length > 0

;(canRun ? describe : describe.skip)(`every wrong projection, with its evidence (${VARIANT_ID})`, () => {
    it("prints them", async () => {
        const byConfidence: Record<string, number> = {}
        const trueByConfidence: Record<string, number> = {}
        let total = 0

        for (const fixture of withReferences) {
            const result = await runFixture({ fixtureId: fixture.id, fixturePath: fixture.absolutePath, variant: variant!, mode: "max" })
            const score = scoreDetection(await replayDetection(result), fixture.expected)

            score.matches.forEach((match) => {
                trueByConfidence[match.detected.confidence] = (trueByConfidence[match.detected.confidence] ?? 0) + 1
            })

            for (const wrong of score.spurious) {
                total++
                const reference = wrong.reference
                byConfidence[reference.confidence] = (byConfidence[reference.confidence] ?? 0) + 1
                // the quote is the span of transcript the matcher built the reference from -
                // which is the evidence for whether the transcript or the matching is at fault
                console.log(`  ${fixture.id.slice(0, 30).padEnd(30)} ${`${reference.book} ${reference.chapter}:${reference.verseStart}`.padEnd(18)} ${reference.confidence.padEnd(6)} from ${JSON.stringify(reference.quote ?? "")}`)
            }
        }

        console.log(`\n  ${total} wrong projections by confidence: ${JSON.stringify(byConfidence)}`)
        console.log(`  correct projections by confidence:      ${JSON.stringify(trueByConfidence)}`)
        console.log(`\n  if the wrong ones cluster in a confidence band the right ones do not, gating on it`)
        console.log(`  raises precision directly. If they share a band, gating only costs recall.`)
        expect(total).toBeGreaterThanOrEqual(0)
    }, 3_600_000)
})
