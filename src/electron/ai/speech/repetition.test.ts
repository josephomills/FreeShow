import { describe, expect, it } from "vitest"
import { findRepeatedTail, hasRepeatedTail, summarizeRepetition } from "./repetition"

const times = (phrase: string, count: number) => Array.from({ length: count }, () => phrase).join(" ")

describe("phrase repetition", () => {
    it("catches the loop seen in a live transcript", () => {
        // no single word here is more than a quarter of the text, so a token-frequency test does
        // not see it. The observed loops ran to twenty repeats and beyond
        expect(hasRepeatedTail(`he spoke to them and ${times("and he saith the LORD", 6)}`)).toBe(true)
    })

    it("catches a single word cycling", () => {
        expect(hasRepeatedTail(times("the king of", 8))).toBe(true)
    })

    it("keeps the first occurrence and marks where the cycle starts", () => {
        const text = `he said ${times("the king of", 6)}`
        const at = findRepeatedTail(text)
        expect(at).toBeGreaterThan(0)
        expect(text.slice(0, at).trim()).toBe("he said the king of")
    })

    it("leaves rhetorical repetition alone", () => {
        // preaching repeats deliberately, and clearing the decoder for it corrupts the sentence
        // that follows. This is the failure that set the thresholds where they are.
        expect(hasRepeatedTail("change your atmosphere atmosphere atmosphere")).toBe(false)
        expect(hasRepeatedTail("holy holy holy is the Lord God almighty")).toBe(false)
        expect(hasRepeatedTail(times("praise the Lord", 3))).toBe(false)
        expect(hasRepeatedTail("Jesus Jesus Jesus Jesus")).toBe(false)
    })

    it("ignores a repeat the speaker has already moved on from", () => {
        // only a tail matters - a live decoder loops at the point it has reached
        expect(hasRepeatedTail(`${times("amen", 10)} and now let us turn to the scripture together`)).toBe(false)
    })

    it("is untroubled by punctuation and case", () => {
        expect(hasRepeatedTail(times("Saith the LORD,", 5))).toBe(true)
    })

    it("says nothing about short text", () => {
        expect(hasRepeatedTail("")).toBe(false)
        expect(hasRepeatedTail("the king of")).toBe(false)
    })
})

describe("summarizing a whole transcript", () => {
    it("reports the share lost to cycles", () => {
        const summary = summarizeRepetition(`he opened the book and ${times("and he saith the LORD", 6)} then closed it`)
        expect(summary.runs).toBe(1)
        expect(summary.share).toBeGreaterThan(0.5)
    })

    it("counts separate cycles separately", () => {
        const summary = summarizeRepetition(`${times("amen", 9)} and later ${times("the king of", 6)} the end`)
        expect(summary.runs).toBe(2)
    })

    it("reports nothing for ordinary speech", () => {
        const summary = summarizeRepetition("and he went up to the mountain to pray alone that evening")
        expect(summary.loopedWords).toBe(0)
        expect(summary.share).toBe(0)
    })

    it("reports nothing for a preacher repeating himself for effect", () => {
        const summary = summarizeRepetition("you must change your atmosphere atmosphere atmosphere because it matters")
        expect(summary.loopedWords).toBe(0)
    })
})
