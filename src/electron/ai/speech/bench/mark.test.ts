// AI BENCH - turn mined sermon clips into detection fixtures with hand-checkable timings.
//
// Not a test. It lives here because marking needs the REAL reference detector to decide where a
// spoken reference begins and ends, and that is TypeScript the harness already imports - running it
// through vitest is cheaper than building a second entry point for it. Gated on AI_MARK so a normal
// run never touches audio.
//
//   AI_MARK=/path/to/hits.json npx vitest run --config config/testing/vitest.config.ts \
//     src/electron/ai/speech/bench/mark.test.ts
//
// phraseEndMs is what every detection-latency number hangs off, so it comes from whisper large-v3
// WORD timestamps, not from a segment boundary: the difference between "the segment containing the
// reference ended" and "the last word of the reference was spoken" is often a second or more.

import { execFileSync } from "child_process"
import fs from "fs"
import os from "os"
import path from "path"
import { describe, expect, it } from "vitest"
import { normalizeSpokenNumbers } from "../../commands/spokenNumbers"
import { detectExplicitReferences } from "../../scripture/detection/references"
import { BENCH_BOOKS } from "./detection"
import { resolveFixtureRoot } from "./fixtures"

const HITS = process.env.AI_MARK
const WHISPER = "/opt/homebrew/bin/whisper-cli"
/** A spoken reference is at most this many words ("first corinthians chapter thirteen verse four"). */
const MAX_REFERENCE_WORDS = 9

interface Word {
    text: string
    from: number
    to: number
}

function userData(): string {
    const home = os.homedir()
    return process.platform === "darwin" ? path.join(home, "Library", "Application Support", "FreeShow") : path.join(process.env.APPDATA || path.join(home, ".config"), "FreeShow")
}

function wordsOf(wav: string, model: string): Word[] {
    const base = wav.replace(/\.wav$/, "-words")
    execFileSync(WHISPER, ["-m", model, "-f", wav, "-l", "en", "-ojf", "-ml", "1", "-sow", "-of", base, "-np", "-nf"], { stdio: "ignore" })
    const json = JSON.parse(fs.readFileSync(`${base}.json`, "utf8"))
    return json.transcription.map((s: any) => ({ text: String(s.text).trim(), from: s.offsets.from, to: s.offsets.to })).filter((w: Word) => w.text)
}

/**
 * The FULLEST reference the speaker actually said.
 *
 * The first window that resolves is not it. "Hebrews chapter 10" is a complete, high-confidence
 * reference to Hebrews 10:1 - but the preacher went on to say "verse 35", and marking the short
 * form made the ground truth wrong in exactly the case detection has to get right. It scored a
 * premature projection of 10:1 as a hit and the correct projection of 10:35 as a miss, which is
 * the opposite of the truth.
 *
 * So: find where a reference starts, then keep extending while it still resolves to the same book
 * and chapter, and take the last window that does. That grows "Hebrews chapter 10" into "Hebrews
 * chapter 10 verse 35" and stops before the next sentence drags in something unrelated.
 */
function referencesIn(words: Word[]) {
    const found: { book: number; chapter: number; verseStart: number; verseEnd?: number; phrase: string; phraseEndMs: number }[] = []
    let cursor = 0

    const resolve = (from: number, to: number) => {
        const text = words
            .slice(from, to)
            .map((w) => w.text)
            .join(" ")
        const refs = detectExplicitReferences(normalizeSpokenNumbers(text), BENCH_BOOKS)
        return refs.length && refs[0].confidence === "high" ? refs[0] : null
    }

    while (cursor < words.length) {
        let start = -1
        let best: { end: number; ref: NonNullable<ReturnType<typeof resolve>> } | null = null

        for (let end = cursor + 1; end <= Math.min(words.length, cursor + MAX_REFERENCE_WORDS); end++) {
            const ref = resolve(cursor, end)
            if (!ref) continue
            if (start < 0) start = cursor
            // keep the longest form of the SAME passage; a different book or chapter is the next
            // reference, not a refinement of this one
            if (best && (ref.bookNumber !== best.ref.bookNumber || ref.chapter !== best.ref.chapter)) break
            best = { end, ref }
        }

        if (!best) {
            cursor++
            continue
        }

        // "verse number twelve" is a way of saying verse 12, not a reference to Numbers. The
        // production guard checks the word before the book name, which the window slicing here
        // hides - the window starts at "number", so "verse" is never in it.
        const previous = words[cursor - 1]?.text.toLowerCase().replace(/[^a-z]/g, "")
        if (best.ref.bookNumber === 4 && (previous === "verse" || previous === "verses")) {
            cursor = best.end
            continue
        }

        found.push({
            book: best.ref.bookNumber,
            chapter: best.ref.chapter,
            verseStart: best.ref.verseStart,
            ...(best.ref.verseEnd !== best.ref.verseStart ? { verseEnd: best.ref.verseEnd } : {}),
            phrase: words
                .slice(cursor, best.end)
                .map((w) => w.text)
                .join(" "),
            phraseEndMs: words[best.end - 1].to
        })
        cursor = best.end
    }
    return found
}

/**
 * The coordinator suppresses an intersecting reference emitted within refCooldownSeconds, so a
 * passage named twice inside that window should reach the screen ONCE. Listing both would score
 * correct behaviour as a miss.
 */
function collapseByCooldown<T extends { book: number; chapter: number; verseStart: number; phraseEndMs: number }>(refs: T[], cooldownMs = 90_000): T[] {
    const kept: T[] = []
    for (const ref of refs.sort((a, b) => a.phraseEndMs - b.phraseEndMs)) {
        const shadowed = kept.some((k) => k.book === ref.book && k.chapter === ref.chapter && k.verseStart === ref.verseStart && ref.phraseEndMs - k.phraseEndMs < cooldownMs)
        if (!shadowed) kept.push(ref)
    }
    return kept
}

const describeIfMarking = HITS && fs.existsSync(HITS) ? describe : describe.skip

describeIfMarking("mark mined clips as detection fixtures", () => {
    it("writes a manifest", () => {
        const { start, duration, hits } = JSON.parse(fs.readFileSync(HITS!, "utf8"))
        const model = path.join(userData(), "bin", "whisper", "models", "ggml-large-v3.bin")
        expect(fs.existsSync(model)).toBe(true)

        const root = resolveFixtureRoot()
        const outDir = path.join(root, "refs")
        fs.mkdirSync(outDir, { recursive: true })

        const fixtures: any[] = []
        for (const [index, hit] of hits.entries()) {
            const id = `${String(index + 1).padStart(2, "0")}-${path
                .basename(hit.file, path.extname(hit.file))
                .toLowerCase()
                .replace(/[^a-z0-9]+/g, "-")
                .slice(0, 40)}`
            const wav = path.join(outDir, `${id}.wav`)
            if (!fs.existsSync(wav)) {
                execFileSync("ffmpeg", ["-y", "-v", "error", "-ss", String(start), "-t", String(duration), "-i", hit.file, "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", wav], { stdio: "ignore" })
            }

            const words = wordsOf(wav, model)
            const expected = collapseByCooldown(referencesIn(words))
            const transcript = words.map((w) => w.text).join(" ")

            console.log(`  ${id}: ${expected.length ? expected.map((r) => `${r.phrase} -> ${r.book}.${r.chapter}:${r.verseStart} @${r.phraseEndMs}ms`).join(" | ") : "(none survived marking)"}`)
            if (!expected.length) continue

            fixtures.push({ id, tier: "sermon", file: path.join("refs", `${id}.wav`), durationMs: duration * 1000, transcript, transcriptSource: "whisper.cpp ggml-large-v3", transcriptApproximate: true, timingSource: "generated", expected })
        }

        const manifest = {
            id: "refs-local",
            description: `Sermon excerpts that contain a spoken scripture reference, ${duration}s from ${start}s in. Reference timings come from whisper large-v3 word timestamps; transcripts are a PSEUDO-reference, valid for ranking engines against each other and circular for scoring whisper. Audio and this manifest are private and never committed.`,
            fixtures
        }
        fs.writeFileSync(path.join(root, "refs.json"), JSON.stringify(manifest, null, 4))

        const total = fixtures.reduce((sum, f) => sum + f.expected.length, 0)
        console.log(`\n  ${fixtures.length} fixtures, ${total} expected reference(s) -> ${path.join(root, "refs.json")}`)
        expect(fixtures.length).toBeGreaterThan(0)
    }, 3_600_000)
})
