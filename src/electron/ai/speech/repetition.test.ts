import { describe, expect, it } from "vitest"
import { findRepeatedTail, hasRepeatedTail } from "./repetition"

describe("phrase repetition", () => {
    it("catches the loop seen in a live transcript", () => {
        // the whole point: no single word here is more than a quarter of the text, so a
        // token-frequency test does not see it
        expect(hasRepeatedTail("he spoke to them and he saith the LORD and he saith the LORD and he saith the LORD")).toBe(true)
    })

    it("catches a single word cycling", () => {
        expect(hasRepeatedTail("the king of the king of the king of the king of")).toBe(true)
    })

    it("keeps the first occurrence and marks where the cycle starts", () => {
        const text = "he said the king of the king of the king of"
        const at = findRepeatedTail(text)
        expect(at).toBeGreaterThan(0)
        expect(text.slice(0, at).trim()).toBe("he said the king of")
    })

    it("leaves emphasis alone", () => {
        expect(hasRepeatedTail("it was very very good")).toBe(false)
        expect(hasRepeatedTail("holy holy is the Lord God almighty")).toBe(false)
    })

    it("leaves a phrase repeated twice alone", () => {
        // people do say things twice for effect; three times back to back is the decoder
        expect(hasRepeatedTail("praise the Lord praise the Lord")).toBe(false)
    })

    it("ignores a repeat the speaker has already moved on from", () => {
        // only a tail matters - a live decoder loops at the point it has reached
        expect(hasRepeatedTail("amen amen amen and now let us turn to the scripture together")).toBe(false)
    })

    it("is untroubled by punctuation and case", () => {
        expect(hasRepeatedTail("Saith the LORD, saith the LORD, saith the LORD.")).toBe(true)
    })

    it("says nothing about short text", () => {
        expect(hasRepeatedTail("")).toBe(false)
        expect(hasRepeatedTail("the king of")).toBe(false)
    })
})
