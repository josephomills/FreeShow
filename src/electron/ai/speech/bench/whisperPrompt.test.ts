// AI BENCH - prices whisper's contextual biasing (Phase 3 of the STT plan).
//
// composeBiblePrompt() IS contextual biasing: whisper conditions its decoder on a text prompt,
// which is the one model-side lever any engine here has for biblical vocabulary. Nobody had
// measured what it buys. This runs whisper over the same transcript-bearing fixtures the
// streaming bench uses, prompt on vs prompt off, and reads the delta - overall WER and a second
// WER restricted to the words the prompt exists for (book names + biblical proper nouns).
//
// The answer decides whether decoder biasing is worth pursuing anywhere else: if the delta is
// small on the engine DESIGNED for prompting, a beam-search host chosen for hotword support
// (the Zipformer detour already measured 2/33 vs 20/33 references) has nothing to offer.
//
//   AI_BENCH=1 npx vitest run --config config/testing/vitest.config.ts src/electron/ai/speech/bench/whisperPrompt.test.ts
//
// Needs whisper-cli on PATH (brew install whisper-cpp) and a ggml model at the app's
// whisper-models dir or WHISPER_MODEL=<path>. Results land in test-output/ai-bench/.

import { execFile } from "child_process"
import fs from "fs"
import os from "os"
import path from "path"
import { describe, expect, it } from "vitest"
import { BIBLE_NAMES_RANKED } from "../whisper/bibleVocabulary"
import { composeBiblePrompt } from "../whisper/prompt"
import { findExecutableInPath } from "../whisper/manager"
import { align, normalizeForWer, vocabularyErrorRate } from "./align"
import { BENCH_BOOKS } from "./detection"
import { resolveUserDataDir } from "./engines"
import { availableFixtures, listManifests, loadFixtureSet, type Fixture } from "./fixtures"
import { bootstrapCi } from "./stats"

const WHISPER_TIMEOUT = 600000

function resolveWhisperModel(): string | null {
    const override = process.env.WHISPER_MODEL
    if (override && fs.existsSync(override)) return override
    const appModel = path.join(resolveUserDataDir(), "whisper-models", "ggml-base.en.bin")
    return fs.existsSync(appModel) ? appModel : null
}

const whisperBinary = findExecutableInPath("whisper-cli")
const whisperModel = resolveWhisperModel()
const enabled = !!process.env.AI_BENCH && !!whisperBinary && !!whisperModel
const describeIfReady = enabled ? describe : describe.skip

/** Transcript-bearing fixtures across every manifest, the same clips the streaming bench scores. */
function transcriptFixtures(): { setId: string; fixture: Fixture }[] {
    const out: { setId: string; fixture: Fixture }[] = []
    for (const manifest of listManifests()) {
        const set = loadFixtureSet(manifest)
        for (const fixture of availableFixtures(set)) {
            if (fixture.transcript) out.push({ setId: set.id, fixture })
        }
    }
    return out
}

function runWhisper(wavPath: string, outBase: string, prompt?: string): Promise<string> {
    return new Promise((resolve, reject) => {
        // mirrors WhisperTranscriber.runCliProcess: no temperature fallback, no prints
        const args = ["-m", whisperModel!, "-l", "en", "-f", wavPath, "-oj", "-of", outBase, "-np", "-t", "4", "-nf"]
        if (prompt) args.push("--prompt", prompt)
        execFile(whisperBinary!, args, { timeout: WHISPER_TIMEOUT }, (err) => {
            if (err) return reject(err)
            try {
                const json = JSON.parse(fs.readFileSync(`${outBase}.json`, "utf8"))
                const text = (json.transcription || []).map((segment: { text?: string }) => segment.text || "").join(" ")
                resolve(text)
            } catch (parseErr) {
                reject(parseErr)
            }
        })
    })
}

interface ClipScore {
    setId: string
    id: string
    wer: number
    vocabRate: number | null // null when the reference contains no vocabulary words
    bookRate: number | null
    vocabTotal: number
    bookTotal: number
}

function scoreClip(setId: string, id: string, reference: string, hypothesis: string, vocabulary: Set<string>, bookNames: Set<string>): ClipScore {
    const alignment = align(normalizeForWer(reference), normalizeForWer(hypothesis))
    const vocab = vocabularyErrorRate(alignment, vocabulary)
    const book = vocabularyErrorRate(alignment, bookNames)
    return {
        setId,
        id,
        wer: alignment.wer,
        vocabRate: vocab.total ? vocab.rate : null,
        bookRate: book.total ? book.rate : null,
        vocabTotal: vocab.total,
        bookTotal: book.total
    }
}

const mean = (values: number[]) => values.reduce((a, b) => a + b, 0) / values.length

describeIfReady("whisper prompt A/B (contextual biasing priced)", () => {
    it("measures WER with the bible prompt on vs off", async () => {
        const clips = transcriptFixtures()
        expect(clips.length).toBeGreaterThan(0)

        // score against the SAME normalization the streaming bench uses, so the numbers compare
        const vocabulary = new Set(BIBLE_NAMES_RANKED.flatMap((name) => normalizeForWer(name)))
        const bookNames = new Set(BENCH_BOOKS.flatMap((book) => book.names).flatMap((name) => normalizeForWer(name)))
        const prompt = composeBiblePrompt()

        const workDir = fs.mkdtempSync(path.join(os.tmpdir(), "freeshow-whisper-ab-"))
        const on: ClipScore[] = []
        const off: ClipScore[] = []

        try {
            for (const { setId, fixture } of clips) {
                const base = path.join(workDir, `${setId}-${fixture.id}`)
                const [withPrompt, withoutPrompt] = await Promise.all([runWhisper(fixture.absolutePath, `${base}-on`, prompt), runWhisper(fixture.absolutePath, `${base}-off`)])
                on.push(scoreClip(setId, fixture.id, fixture.transcript!, withPrompt, vocabulary, bookNames))
                off.push(scoreClip(setId, fixture.id, fixture.transcript!, withoutPrompt, vocabulary, bookNames))
            }
        } finally {
            fs.rmSync(workDir, { recursive: true, force: true })
        }

        // paired per-clip deltas (on - off): negative = the prompt helped
        const werDeltas = on.map((clip, i) => clip.wer - off[i].wer)
        const vocabPairs = on.map((clip, i) => ({ on: clip.vocabRate, off: off[i].vocabRate })).filter((pair) => pair.on !== null && pair.off !== null)
        const bookPairs = on.map((clip, i) => ({ on: clip.bookRate, off: off[i].bookRate })).filter((pair) => pair.on !== null && pair.off !== null)

        const werCi = bootstrapCi(werDeltas, mean)
        const lines: string[] = []
        const pct = (value: number) => `${(value * 100).toFixed(1)}%`

        lines.push(`whisper prompt A/B - ${clips.length} clips, model ${path.basename(whisperModel!)}`)
        lines.push(`prompt (${prompt.length} chars): ${prompt}`)
        lines.push("")
        lines.push(`overall WER      on ${pct(mean(on.map((c) => c.wer)))}   off ${pct(mean(off.map((c) => c.wer)))}   delta ${pct(mean(werDeltas))} (CI ${pct(werCi.low)}..${pct(werCi.high)})`)
        if (vocabPairs.length) {
            const deltas = vocabPairs.map((pair) => pair.on! - pair.off!)
            const ci = bootstrapCi(deltas, mean)
            lines.push(`bible-name WER   on ${pct(mean(vocabPairs.map((p) => p.on!)))}   off ${pct(mean(vocabPairs.map((p) => p.off!)))}   delta ${pct(mean(deltas))} (CI ${pct(ci.low)}..${pct(ci.high)})  [${vocabPairs.length} clips, ${on.reduce((a, c) => a + c.vocabTotal, 0)} name tokens]`)
        }
        if (bookPairs.length) {
            const deltas = bookPairs.map((pair) => pair.on! - pair.off!)
            const ci = bootstrapCi(deltas, mean)
            lines.push(`book-name WER    on ${pct(mean(bookPairs.map((p) => p.on!)))}   off ${pct(mean(bookPairs.map((p) => p.off!)))}   delta ${pct(mean(deltas))} (CI ${pct(ci.low)}..${pct(ci.high)})  [${bookPairs.length} clips, ${on.reduce((a, c) => a + c.bookTotal, 0)} book tokens]`)
        }
        lines.push("")
        lines.push("per set (WER on/off):")
        const bySets = new Map<string, { on: number[]; off: number[] }>()
        on.forEach((clip, i) => {
            const entry = bySets.get(clip.setId) || { on: [], off: [] }
            entry.on.push(clip.wer)
            entry.off.push(off[i].wer)
            bySets.set(clip.setId, entry)
        })
        bySets.forEach((entry, setId) => lines.push(`  ${setId.padEnd(12)} ${pct(mean(entry.on))} / ${pct(mean(entry.off))}  (${entry.on.length} clips)`))

        const report = lines.join("\n")
        console.log(report)

        const outDir = path.join(__dirname, "../../../../../test-output/ai-bench")
        fs.mkdirSync(outDir, { recursive: true })
        fs.writeFileSync(path.join(outDir, `whisper-prompt-ab-${new Date().toISOString().replace(/[:.]/g, "-")}.txt`), report)
        fs.writeFileSync(path.join(outDir, `whisper-prompt-ab-latest.json`), JSON.stringify({ prompt, on, off }, null, 2))
    }, 3600000)
})
