import JsonBible from "json-bible"
import { describe, expect, it } from "vitest"
import type { BibleInstance } from "../../components/drawer/bible/scripture"
import { getBookExact, getChapterExact, hasBookNumber } from "./bibleLookup"

// A bible shaped like the real .fsb files, but deliberately missing Psalms (19) so the silent
// fallback json-bible performs on a miss is observable.
function makeBible(bookNumbers: number[]) {
    const NAMES: { [key: number]: string } = { 1: "Genesis", 19: "Psalms", 40: "Matthew" }
    return {
        name: "TEST",
        metadata: {},
        books: bookNumbers.map((number) => ({
            number,
            name: NAMES[number] || `Book ${number}`,
            // Genesis has 50 chapters, Psalms 150 - enough to tell a clamp from a fallback
            chapters: Array.from({ length: number === 19 ? 150 : 50 }, (_, i) => ({
                number: i + 1,
                verses: [{ number: 8, text: `${NAMES[number] || number} ${i + 1}:8` }]
            }))
        }))
    }
}

const loadBible = async (bookNumbers: number[]) => (await JsonBible(makeBible(bookNumbers) as never)) as unknown as BibleInstance

describe("json-bible's silent fallbacks", () => {
    it("resolves a missing book to the FIRST book rather than failing", async () => {
        const bible = await loadBible([1, 40])

        // this is the upstream behavior the helpers exist to contain - if it ever changes, the
        // guards below become redundant rather than wrong, so pin it explicitly
        const raw = await bible.getBook(19)
        expect(raw.name).toBe("Genesis")
        expect(raw.index).toBe(-1)
    })

    it("resolves a missing chapter to chapter 1 but reports the REQUESTED number", async () => {
        const bible = await loadBible([1, 40])

        const Genesis = await bible.getBook(1)
        const raw = await Genesis.getChapter(52) // Genesis has 50

        expect(raw.number).toBe(52) // the label the output would show
        expect(raw.data.number).toBe(1) // the content it would actually show
        expect(raw.index).toBe(-1)
    })
})

describe("getBookExact", () => {
    it("returns null instead of the fallback book when the number is absent", async () => {
        const bible = await loadBible([1, 40])
        expect(await getBookExact(bible, 19)).toBeNull()
    })

    it("returns the requested book when it is present", async () => {
        const bible = await loadBible([1, 19, 40])

        const Book = await getBookExact(bible, 19)
        expect(Book?.name).toBe("Psalms")
        expect(Book?.data.number).toBe(19)
    })
})

describe("getChapterExact", () => {
    it("returns null instead of chapter 1 when the chapter is absent", async () => {
        const bible = await loadBible([1, 19, 40])

        const Genesis = await getBookExact(bible, 1)
        expect(await getChapterExact(Genesis!, 52)).toBeNull()
    })

    it("returns the requested chapter when it is present", async () => {
        const bible = await loadBible([1, 19, 40])

        const Psalms = await getBookExact(bible, 19)
        const Chapter = await getChapterExact(Psalms!, 52)
        expect(Chapter?.data.number).toBe(52)
        expect(Chapter?.data.verses[0]?.text).toBe("Psalms 52:8")
    })

    // the reported failure, end to end: Psalm 52:8 detected, Genesis 52:8 projected carrying
    // Genesis 1:8 text, because both lookups fell back and the label kept the detected numbers
    it("blocks the Psalm 52:8 -> Genesis 1:8 projection", async () => {
        const bible = await loadBible([1, 40])

        const Book = await getBookExact(bible, 19)
        expect(Book).toBeNull()

        // what the unchecked path would have put on the output
        const unchecked = await (await bible.getBook(19)).getChapter(52)
        expect(unchecked.number).toBe(52)
        expect(unchecked.data.verses[0]?.text).toBe("Genesis 1:8")
    })
})

describe("hasBookNumber", () => {
    it("tells a 66 book list that is missing the number from one that has it", () => {
        expect(hasBookNumber(makeBible([1, 40]).books, 19)).toBe(false)
        expect(hasBookNumber(makeBible([1, 19, 40]).books, 19)).toBe(true)
        // book numbers can arrive as strings from stored references
        expect(hasBookNumber(makeBible([1, 19]).books, "19")).toBe(true)
    })
})
