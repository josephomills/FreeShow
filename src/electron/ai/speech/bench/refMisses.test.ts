// AI BENCH - what the engine actually said where a reference was spoken.
//
// Detection can do nothing with a transcript that never contained the reference, so the ceiling on
// recall is set by transcription, not by matching. This prints, for every marked reference, the
// words the engine committed around the moment it was spoken next to the words whisper large-v3
// heard there - which is the only way to tell "detection missed it" from "it was never said to
// detection in the first place".
//
//   AI_BENCH=1 npx vitest run --config config/testing/vitest.config.ts \
//     src/electron/ai/speech/bench/refMisses.test.ts

import { describe, expect, it } from "vitest"
import { normalizeSpokenNumbers } from "../../commands/spokenNumbers"
import { detectExplicitReferences } from "../../scripture/detection/references"
import { BENCH_BOOKS } from "./detection"
import { availableVariants, benchModelReady, hasSherpa } from "./engines"
import { availableFixtures, listManifests, loadFixtureSet, type Fixture } from "./fixtures"
import { runFixture } from "./runner"

const VARIANT_ID = process.env.AI_BENCH_VARIANT || "stream multi-1120"
/** How far back from the end of a spoken phrase to look for what the engine committed. */
const LOOKBACK_MS = 7000
const LOOKAHEAD_MS = 2500

const withReferences: Fixture[] = listManifests()
    .flatMap((manifest) => availableFixtures(loadFixtureSet(manifest)))
    .filter((fixture) => fixture.expected.length > 0)

const variant = availableVariants().find((entry) => entry.id === VARIANT_ID)
const canRun = !!process.env.AI_BENCH && hasSherpa() && !!variant && benchModelReady(variant.modelSet) && withReferences.length > 0

;(canRun ? describe : describe.skip)(`what the engine heard where a reference was spoken (${VARIANT_ID})`, () => {
    it("prints every reference next to the engine's words", async () => {
        let resolvable = 0
        let total = 0

        for (const fixture of withReferences) {
            const result = await runFixture({ fixtureId: fixture.id, fixturePath: fixture.absolutePath, variant: variant!, mode: "max" })

            for (const reference of fixture.expected) {
                total++
                const from = reference.phraseEndMs - LOOKBACK_MS
                const to = reference.phraseEndMs + LOOKAHEAD_MS

                // segments carry the engine's own timings, so this is the window the engine
                // believes those words occupy - not a guess from the audio clock
                const heard = result.events
                    .filter((event) => event.kind === "segment" && event.text && (event.endMs ?? 0) >= from && (event.startMs ?? 0) <= to)
                    .map((event) => event.text)
                    .join(" ")
                    .replace(/\s+/g, " ")
                    .trim()

                const found = detectExplicitReferences(normalizeSpokenNumbers(heard), BENCH_BOOKS)
                const hit = found.some((r) => r.bookNumber === reference.book && r.chapter === reference.chapter && r.verseStart === reference.verseStart)
                if (hit) resolvable++

                console.log(`\n  ${hit ? "OK  " : "MISS"} ${fixture.id.slice(0, 34)}  want ${reference.book}.${reference.chapter}:${reference.verseStart}`)
                console.log(`       said : ${reference.phrase.slice(-80)}`)
                console.log(`       heard: ${heard.slice(-110) || "(nothing in this window)"}`)
                if (!hit && found.length) console.log(`       got  : ${found.map((r) => `${r.bookNumber}.${r.chapter}:${r.verseStart}`).join(", ")}`)
            }
        }

        console.log(`\n  ${resolvable}/${total} references survive transcription well enough for detection to resolve them`)
        console.log(`  that fraction is the ceiling on recall - no change to matching can raise it`)
        expect(total).toBeGreaterThan(0)
    }, 3_600_000)
})
