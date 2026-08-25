// AI BENCH - what the streaming driver does when audio goes missing.
//
// Its whole premise is that the audio it receives is continuous: the encoder cache is only valid if
// nothing is missing, and it feeds every chunk including silence to keep that true. A live fault
// appeared while the same laptop was decoding video, running the app and running the engine, which
// is exactly when the capture worklet or the IPC hop would start dropping frames. The batch driver
// it replaced was immune - it re-decoded each utterance from scratch, so a gap was only a gap.
//
//   AI_BENCH=1 npx vitest run --config config/testing/vitest.config.ts \
//     src/electron/ai/speech/bench/dropped.test.ts

import { describe, expect, it } from "vitest"
import { summarizeRepetition } from "../repetition"
import { availableVariants, benchModelReady, hasSherpa } from "./engines"
import { availableFixtures, listManifests, loadFixtureSet, type Fixture } from "./fixtures"
import { runFixture } from "./runner"

const VARIANT_ID = process.env.AI_BENCH_VARIANT || "stream multi-1120"
const DROP_RATES = [0, 0.01, 0.05, 0.15]

// AI_BENCH_FIXTURE names one; otherwise the longest available, because a cache that has drifted
// out of step with the audio needs minutes to show what it does
const candidates = listManifests().flatMap((manifest) => availableFixtures(loadFixtureSet(manifest)))
const fixture: Fixture | undefined = process.env.AI_BENCH_FIXTURE ? candidates.find((entry) => entry.id === process.env.AI_BENCH_FIXTURE) : [...candidates].sort((a, b) => (b.durationMs ?? 0) - (a.durationMs ?? 0))[0]

const variant = availableVariants().find((entry) => entry.id === VARIANT_ID)
const canRun = !!process.env.AI_BENCH && hasSherpa() && !!variant && benchModelReady(variant.modelSet) && !!fixture

;(canRun ? describe : describe.skip)(`dropped audio (${VARIANT_ID})`, () => {
    it("reports what missing chunks do to the transcript", async () => {
        console.log(`\n  ${fixture!.id} (${((fixture!.durationMs || 0) / 60000).toFixed(1)} min)`)
        console.log(`  ${"dropped".padStart(8)}${"words".padStart(8)}${"looped".padStart(8)}${"share".padStart(8)}${"runs".padStart(6)}${"longest".padStart(9)}`)

        for (const dropRate of DROP_RATES) {
            const result = await runFixture({ fixtureId: fixture!.id, fixturePath: fixture!.absolutePath, variant: variant!, mode: "max", dropRate })
            const repetition = summarizeRepetition(result.hypothesis)
            console.log(`  ${`${(dropRate * 100).toFixed(0)}%`.padStart(8)}${String(repetition.totalWords).padStart(8)}${String(repetition.loopedWords).padStart(8)}${`${(repetition.share * 100).toFixed(1)}%`.padStart(8)}${String(repetition.runs).padStart(6)}${String(repetition.longestRun).padStart(9)}`)
        }

        expect(true).toBe(true)
    }, 3_600_000)
})
