import { describe, expect, it } from "vitest"
import { isAnnotationOnly, isMusicAnnotation } from "./annotations"

describe("non-speech labels", () => {
    it("recognises the labels the streaming engine actually emits", () => {
        // both seen in a live session while a worship set was playing
        expect(isMusicAnnotation("[MUSIC PLAYING]")).toBe(true)
        expect(isMusicAnnotation("(upbeat music)")).toBe(true)
    })

    it("recognises whisper's sung-content marker", () => {
        expect(isMusicAnnotation("♪ la la la ♪")).toBe(true)
    })

    it("recognises other non-speech labels without calling them music", () => {
        expect(isAnnotationOnly("[BLANK_AUDIO]")).toBe(true)
        expect(isAnnotationOnly("*applause*")).toBe(true)
        expect(isMusicAnnotation("[BLANK_AUDIO]")).toBe(false)
    })

    it("leaves speech alone", () => {
        expect(isAnnotationOnly("turn with me to Ephesians chapter two")).toBe(false)
        expect(isMusicAnnotation("turn with me to Ephesians chapter two")).toBe(false)
    })

    it("does not call a sentence music because it mentions music", () => {
        // the label has to BE the annotation, not a word inside a sentence
        expect(isMusicAnnotation("the music ministry will lead us now")).toBe(false)
    })

    it("still recognises a label that carries speech beside it", () => {
        expect(isMusicAnnotation("(upbeat music) and welcome everybody")).toBe(true)
        expect(isAnnotationOnly("(upbeat music) and welcome everybody")).toBe(false)
    })
})

describe("a label split across segments", () => {
    it("recognises the opening half", () => {
        // the streaming driver commits whole words, so "[MUSIC PLAYING]" can arrive as "[MUSIC"
        // and then "PLAYING]" - flagging only the half with the bracket would let the other half
        // reach scripture detection as speech
        expect(isMusicAnnotation("[MUSIC")).toBe(true)
        expect(isMusicAnnotation("(upbeat music")).toBe(true)
    })

    it("does not let a stray bracket swallow a sentence", () => {
        expect(isMusicAnnotation("the music ministry (led by Grace) will come now")).toBe(false)
    })
})
