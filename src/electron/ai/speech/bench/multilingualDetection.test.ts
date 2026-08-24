import { describe, expect, it } from "vitest"
import { normalizeSpokenNumbers } from "../../commands/spokenNumbers"
import { detectExplicitReferences } from "../../scripture/detection/references"
import { BENCH_BOOKS } from "./detection"

/**
 * The two exports write numbers differently, and scripture detection parses references out of the
 * transcript - so a model swap that broke number parsing would stop the feature finding anything
 * while the transcript still looked fine on screen. That is the worst failure mode available here,
 * which is why it gets its own test rather than being assumed from the WER numbers.
 *
 * Both strings below are VERBATIM output from the real models on the same sermon audio.
 */
describe("scripture detection across the two Nemotron exports", () => {
    // english-only export: digits
    const english = "do a bible study together just using this outline matthew 6 33 please matthew 6 33"
    // multilingual 3.5 export: numbers spelled out, and it mis-hears the book name once
    const multilingual = "I've used stari together just using this outline Matthew six thirty three Matthe six thirty three man with a dictionary"

    const detect = (text: string) => detectExplicitReferences(normalizeSpokenNumbers(text), BENCH_BOOKS)

    it("finds Matthew 6:33 in the English-only export's output", () => {
        const found = detect(english)
        console.log(`  english      -> ${JSON.stringify(found.map((r) => `${r.book} ${r.chapter}:${r.verseStart} (${r.confidence})`))}`)
        expect(found.some((r) => r.bookNumber === 40 && r.chapter === 6 && r.verseStart === 33)).toBe(true)
    })

    it("finds Matthew 6:33 in the multilingual export's spelled-out output", () => {
        const found = detect(multilingual)
        console.log(`  multilingual -> ${JSON.stringify(found.map((r) => `${r.book} ${r.chapter}:${r.verseStart} (${r.confidence})`))}`)
        expect(found.some((r) => r.bookNumber === 40 && r.chapter === 6 && r.verseStart === 33)).toBe(true)
    })
})
