// AI AUTO SCRIPTURE
// tier 1: fast local regex detection of explicitly spoken references ("John chapter 3 verse 16")
// tier 2: LLM detection over the rolling transcript for paraphrased/quoted references (optional, needs an API key)

import type { AiScriptureBook, AiScriptureTranslation, AiScriptureState, DetectedReference } from "../../../../types/ai/AiScripture"
import { normalizeSpokenNumbers } from "../../commands/spokenNumbers"
import { maxVerseInChapter } from "../chapterVerseCounts"
import { LLM_API_TIMEOUT } from "../../llm/models/APIModel"
import { getLLMScriptureProvider } from "../llmTalkScripture"
import type { BookIndex } from "./references"
import { buildBookIndex, matchReferences } from "./references"

// COORDINATOR

// the anchor passage: what is live on the output right now - bare "verse N" mentions resolve against it
export interface AiScriptureAnchor {
    book: string
    bookNumber: number
    chapter: number
    verseStart: number
    verseEnd: number
}

// cue-gated: the word "verse(s)" with the number AFTER it is required, so "he has fifteen verses" never fires
// "verse 5", "verse number 5", "the 5th verse" - a range may follow ("verses 3 to 5", "3 and 4");
// "and" as a range word is safe here because the verse word is present by construction
const BARE_VERSE_REGEX = /(^|[^a-z0-9])((?:the\s+)?(?:verses?\s+(?:number\s+)?(?<n1>\d{1,3})\b|(?<n2>\d{1,3})(?:st|nd|rd|th)\s+verses?\b)(?:\s*(?:-|–|to\b|through\b|and\b|till\b|until\b)\s*(?<end>\d{1,3})\b)?)/g

// a bare number while a passage is live: "...with all boldness. Thirty one. And when they had
// prayed..." - reading straight through, the next verse is cued by its number alone
const BARE_NUMBER_REGEX = /(^|[^a-z0-9:])(\d{1,3})\b(?!\s*:)/g

interface TranscriptSegment {
    text: string
    startMs: number
    endMs: number
}

interface EmittedReference {
    book: string
    chapter: number
    verseStart: number
    verseEnd: number
    timestamp: number
}

interface DetectionCandidate {
    book: string
    bookNumber: number
    chapter: number
    verseStart: number
    verseEnd: number
    confidence: "high" | "medium" | "low"
    type: "explicit" | "quoted"
    quote?: string
    spokenBibleId?: string
}

interface DetectionCoordinatorOptions {
    books: AiScriptureBook[]
    /** Installed translations, so a version named with the reference resolves to its bible. */
    translations?: AiScriptureTranslation[]
    llm: { provider: string; model: string } | null
    getApiKey: (providerId: string) => string
    onDetection: (ref: DetectedReference) => void
    onStatus: (state: AiScriptureState, extra?: { message?: string; keyless?: boolean }) => void
    cooldownSeconds?: number
    /**
     * Hold a reference sitting at the very end of the transcript until more speech follows it or
     * the utterance closes, because it may still be being spoken. On unless explicitly disabled;
     * the switch exists so the benchmark can measure what the guard costs and what it buys on the
     * same audio, rather than against a remembered number.
     */
    holdProvisionalReferences?: boolean
}

const ROLLING_MAX_MS = 90000 // rolling transcript cap
const ROLLING_MAX_CHARS = 2000
const TIER1_WINDOW_MS = 15000 // tier 1 only rescans the most recent speech
const LLM_MIN_NEW_WORDS = 15 // don't call the LLM again until this much new speech arrived
const LLM_ALREADY_DETECTED_MS = 180000 // recently emitted refs sent to the LLM so it skips them
const DEFAULT_COOLDOWN_SECONDS = 90 // suppress re-emitting an intersecting reference within this window

/**
 * How long a chapter with no spoken verse waits to see whether one follows.
 *
 * "Ephesians chapter 2" is a complete reference to Ephesians 2:1, and preachers say it on the way
 * to "verse 8" constantly - so projecting it immediately put the wrong passage on screen and then
 * corrected itself. Verse 1 is a DEFAULT here, not something anyone said, which is what separates
 * this from a reference the speaker finished.
 *
 * Measured in audio time, so it does not depend on how fast the machine is. A preacher who really
 * did mean the whole chapter waits this long to see it; one who is mid-reference is not
 * contradicted on screen.
 */
const BARE_CHAPTER_HOLD_MS = 4000

export class DetectionCoordinator {
    private opts: DetectionCoordinatorOptions
    private bookIndex: BookIndex
    private cooldownMs: number
    private holdProvisional: boolean

    // a single replaced anchor object - strictly bounded, never accumulates over a long sermon
    private anchor: AiScriptureAnchor | null = null
    private anchorBookPrefix: RegExp | null

    private segments: TranscriptSegment[] = []
    /** Chapters seen with no spoken verse, waiting to learn whether one follows. Keyed book.chapter. */
    private pendingChapters = new Map<string, { candidate: DetectionCandidate; atMs: number }>()
    private emitted = new Map<string, EmittedReference[]>() // key: "bookNumber.chapter"
    private idCounter = 0
    private stopped = false

    // tier 2 state
    private totalWords = 0
    private wordsAtLastLlmCall = 0
    private llmController: AbortController | null = null
    private llmCallStartedAt = 0
    private llmRerunPending = false // new speech arrived while a call was in flight: re-run once it settles
    private llmStopped = false
    private llmPermanentFailures = 0 // consecutive permanent-class errors - two stop tier 2
    private llmCooldownUntil = 0

    constructor(opts: DetectionCoordinatorOptions) {
        this.opts = opts
        this.bookIndex = buildBookIndex(opts.books, opts.translations)
        this.anchorBookPrefix = this.bookIndex.bookPattern ? new RegExp("(?:^|[^a-z0-9])(?:" + this.bookIndex.bookPattern + ")[,.]?\\s+$") : null
        this.cooldownMs = (opts.cooldownSeconds ?? DEFAULT_COOLDOWN_SECONDS) * 1000
        this.holdProvisional = opts.holdProvisionalReferences !== false
    }

    // replace the anchor passage (what is live on the output right now)
    /** The provider was (re)configured mid-session - arm tier 2 fresh, clearing any pause/backoff. */
    updateLlm(llm: DetectionCoordinatorOptions["llm"]): void {
        this.opts.llm = llm
        this.llmStopped = false
        this.llmPermanentFailures = 0
        this.llmCooldownUntil = 0
    }

    /** The Search Bibles selection changed mid-session - the spoken book-name index follows it. */
    updateBooks(books: AiScriptureBook[]): void {
        this.opts.books = books
        this.bookIndex = buildBookIndex(books, this.opts.translations)
        this.anchorBookPrefix = this.bookIndex.bookPattern ? new RegExp("(?:^|[^a-z0-9])(?:" + this.bookIndex.bookPattern + ")[,.]?\\s+$") : null
    }

    updateContext(ctx: AiScriptureAnchor): void {
        this.anchor = ctx
    }

    onTranscriptSegment(segment: { text: string; startMs: number; endMs: number; utteranceEnd?: boolean }): void {
        if (this.stopped) return

        this.segments.push(segment)
        this.totalWords += countWords(segment.text)
        this.trimRollingTranscript()

        this.runTier1(segment.utteranceEnd === true)
        this.maybeRunTier2()
    }

    stop(): void {
        // a chapter still waiting for a verse that never came is what the speaker meant after all;
        // the wait only ends when speech does, and this is where speech ends
        this.pendingChapters.forEach((pending) => this.tryEmit(pending.candidate, "regex"))
        this.pendingChapters.clear()

        this.stopped = true
        this.llmRerunPending = false
        if (this.llmController) {
            this.llmController.abort()
            this.llmController = null
        }
        this.segments = []
        this.emitted.clear()
        this.pendingChapters.clear()
    }

    // TIER 1

    /**
     * `settled` means the transcriber closed the utterance, so the newest words are the last ones
     * of that utterance and nothing will extend them. Until then a reference sitting at the very
     * end of the transcript is only PROVISIONALLY complete - see the tailAnchored guard below.
     */
    private runTier1(settled: boolean) {
        const newestEnd = this.segments[this.segments.length - 1].endMs
        const windowText = this.segments
            .filter((segment) => segment.endMs >= newestEnd - TIER1_WINDOW_MS)
            .map((segment) => segment.text)
            .join(" ")

        const newestMs = this.segments[this.segments.length - 1].endMs
        const matches = matchReferences(windowText, this.bookIndex)

        // a chapter the speaker went on to give a verse for is not a bare chapter any more
        matches.forEach((match) => {
            if (!match.bareChapter) this.pendingChapters.delete(`${match.bookNumber}.${match.chapter}`)
        })

        matches.forEach((match) => {
            // A reference at the very end of the transcript may still be being spoken. Emitting it
            // immediately is why "Matthew 6:33" reached the screen as Matthew 6:1, then 6:30, then
            // 6:33 - three passages in just over a second, two of them wrong, because "matthew 6"
            // and "matthew 6 30" are each a valid reference on their own. Holding it costs one
            // segment of latency (the streaming engine emits several a second) and only until the
            // speaker says anything else at all, including the pause that closes the utterance.
            if (this.holdProvisional && match.tailAnchored && !settled) return

            const candidate: DetectionCandidate = { book: match.book, bookNumber: match.bookNumber, chapter: match.chapter, verseStart: match.verseStart, verseEnd: match.verseEnd, confidence: match.confidence, type: "explicit", quote: match.quote, spokenBibleId: match.spokenBibleId }

            // A chapter with no spoken verse waits to learn whether one follows - see
            // BARE_CHAPTER_HOLD_MS. Verse 1 is this reference's default, not the speaker's word.
            if (this.holdProvisional && match.bareChapter) {
                const key = `${match.bookNumber}.${match.chapter}`
                if (!this.pendingChapters.has(key)) this.pendingChapters.set(key, { candidate, atMs: newestMs })
                return
            }

            this.tryEmit(candidate, "regex")
        })

        this.flushPendingChapters(newestMs)

        this.runAnchorTier1(windowText, settled)
    }

    /** Emit chapters whose wait is up - nobody gave them a verse, so the chapter is what was meant. */
    private flushPendingChapters(newestMs: number) {
        for (const [key, pending] of this.pendingChapters) {
            if (newestMs - pending.atMs < BARE_CHAPTER_HOLD_MS) continue
            this.pendingChapters.delete(key)
            this.tryEmit(pending.candidate, "regex")
        }
    }

    // bare "verse N" / "verses N to M" mentions (no book named) resolve against the anchor passage
    private runAnchorTier1(windowText: string, settled: boolean) {
        const anchor = this.anchor
        if (!anchor) return

        const normalized = normalizeSpokenNumbers(windowText)

        // spans already matched as full references - the "verse 16" inside "john chapter 3 verse 16" is not anchor-relative
        const covered: [number, number][] = []
        if (this.bookIndex.regex) {
            this.bookIndex.regex.lastIndex = 0
            let full: RegExpExecArray | null
            while ((full = this.bookIndex.regex.exec(normalized)) !== null) covered.push([full.index, full.index + full[0].length])
        }

        BARE_VERSE_REGEX.lastIndex = 0
        let match: RegExpExecArray | null
        while ((match = BARE_VERSE_REGEX.exec(normalized)) !== null) {
            const start = match.index + match[1].length
            if (covered.some(([from, to]) => start >= from && start < to)) continue
            // a book name directly before makes it a partial explicit reference ("john verse 7"), not an anchor-relative one
            if (this.anchorBookPrefix?.test(normalized.slice(0, start))) continue

            const groups = (match.groups || {}) as { n1?: string; n2?: string; end?: string }
            const verseStart = parseInt(groups.n1 ?? groups.n2 ?? "", 10)
            if (!(verseStart >= 1)) continue
            let verseEnd = groups.end !== undefined ? parseInt(groups.end, 10) : verseStart
            if (verseEnd < verseStart) verseEnd = verseStart

            // same growing-number problem as a full reference: "verse 3" becomes "verse 33"
            if (this.holdProvisional && !settled && !normalized.slice(match.index + match[0].length).trim()) continue

            // the anchor is the chapter live on screen, so a bare verse mention is context-certain
            this.tryEmit({ book: anchor.book, bookNumber: anchor.bookNumber, chapter: anchor.chapter, verseStart, verseEnd, confidence: "high", type: "explicit", quote: match[2] }, "regex")
        }

        // Reading straight through, the next verse is often cued by its number ALONE: "...speak
        // your word with all boldness. Thirty one. And when they had prayed..." (a live service,
        // reading Acts 4 - verse 31 was only caught later by the quote matcher). Only the number
        // that is EXACTLY the next verse after the live one counts, and only while that verse
        // exists: any other number in preaching is a count, an age, an amount - never this one.
        const nextVerse = anchor.verseEnd + 1
        if (nextVerse <= (maxVerseInChapter(anchor.bookNumber, anchor.chapter) || Infinity)) {
            BARE_NUMBER_REGEX.lastIndex = 0
            while ((match = BARE_NUMBER_REGEX.exec(normalized)) !== null) {
                if (parseInt(match[2], 10) !== nextVerse) continue
                const start = match.index + match[1].length
                if (covered.some(([from, to]) => start >= from && start < to)) continue
                // "verse 31"/"chapter 31" carry their own word and their own rules
                if (/(?:verses?|chapters?|numbers?)\s*$/.test(normalized.slice(0, start))) continue
                if (this.anchorBookPrefix?.test(normalized.slice(0, start))) continue
                // the number may still be growing ("31" on its way to "310")
                if (this.holdProvisional && !settled && !normalized.slice(match.index + match[0].length).trim()) continue

                this.tryEmit({ book: anchor.book, bookNumber: anchor.bookNumber, chapter: anchor.chapter, verseStart: nextVerse, verseEnd: nextVerse, confidence: "high", type: "explicit", quote: match[2] }, "regex")
            }
        }
    }

    // TIER 2 - single flight, newest transcript wins once the in-flight call settles

    private maybeRunTier2() {
        const llm = this.opts.llm
        if (!llm || this.llmStopped) return

        // a call is in flight: never abort it just because new speech arrived (during continuous speech that would starve
        // tier 2 completely) - mark a re-run so the fresh transcript is analyzed as soon as the call settles.
        // only calls stuck past the request timeout are aborted & replaced right away
        if (this.llmController) {
            if (Date.now() - this.llmCallStartedAt <= LLM_API_TIMEOUT) {
                this.llmRerunPending = true
                return
            }
            this.llmController.abort()
            this.llmController = null
        }

        if (Date.now() < this.llmCooldownUntil) return
        if (this.totalWords - this.wordsAtLastLlmCall < LLM_MIN_NEW_WORDS) return

        const apiKey = this.opts.getApiKey(llm.provider)
        if (!apiKey) return

        this.wordsAtLastLlmCall = this.totalWords
        const controller = new AbortController()
        this.llmController = controller
        this.llmCallStartedAt = Date.now()

        const transcript = this.segments.map((segment) => segment.text).join(" ")
        const liveContext = this.anchor ? "Live on screen: " + this.anchor.book + " " + this.anchor.chapter + ":" + this.anchor.verseStart + "-" + this.anchor.verseEnd + ". Bare verse mentions likely refer to this passage." : undefined
        Promise.resolve()
            .then(() => getLLMScriptureProvider(llm.provider as any).detectScripture(apiKey, llm.model, { transcript, alreadyDetected: this.recentDetectionStrings(), liveContext }, controller.signal))
            .then(
                (result: any) => {
                    if (this.llmController !== controller) return // aborted/superseded
                    this.llmController = null
                    this.llmPermanentFailures = 0 // a working provider clears the strike count
                    this.handleLlmReferences(Array.isArray(result?.references) ? result.references : [])
                    this.runPendingRerun()
                },
                (err: any) => {
                    if (this.llmController !== controller) return // aborted/superseded
                    this.llmController = null
                    this.handleLlmError(err)
                    this.runPendingRerun()
                }
            )
    }

    // speech arrived while the last call was in flight: immediately analyze the fresh transcript
    // (maybeRunTier2 re-checks the cooldown/new-words conditions & whether tier 2 was stopped meanwhile)
    private runPendingRerun() {
        if (!this.llmRerunPending) return
        this.llmRerunPending = false
        if (!this.stopped) this.maybeRunTier2()
    }

    private handleLlmReferences(references: any[]) {
        references.forEach((raw: any) => {
            const chapter = Math.floor(Number(raw?.chapter))
            const verseStart = Math.floor(Number(raw?.verseStart))
            if (!Number.isFinite(chapter) || chapter < 1 || !Number.isFinite(verseStart) || verseStart < 1) return

            let verseEnd = Math.floor(Number(raw?.verseEnd))
            if (!Number.isFinite(verseEnd) || verseEnd < verseStart) verseEnd = verseStart

            // resolve the book against the active book table first, fall back to the canon book number
            const rawName = String(raw?.book || "").trim()
            const nameMatch = this.bookIndex.byToken.get(rawName.toLowerCase().replace(/\s+/g, " "))
            const bookNumber = nameMatch ? nameMatch.number : Math.floor(Number(raw?.bookNumber))
            if (!nameMatch && (!Number.isFinite(bookNumber) || bookNumber < 1 || bookNumber > 66)) return

            const confidence: "high" | "medium" | "low" = raw?.confidence === "high" || raw?.confidence === "low" ? raw.confidence : "medium"
            const type: "explicit" | "quoted" = raw?.type === "quoted" ? "quoted" : "explicit"
            const quote = typeof raw?.quote === "string" && raw.quote ? raw.quote : undefined

            this.tryEmit({ book: nameMatch ? nameMatch.name : rawName, bookNumber, chapter, verseStart, verseEnd, confidence, type, quote }, "llm")
        })
    }

    private handleLlmError(err: any) {
        const code = err?.code

        // permanent-class errors will not fix themselves - bad key, unknown model, malformed
        // request. But providers occasionally return one as a transient blip, and a single
        // misclassified error must not kill tier 2 for a whole service - it takes TWO in a row
        // (a genuinely broken setup repeats on the very next window anyway). Tier 1 keeps running
        if (code === "invalid_key" || code === "forbidden" || code === "model_not_found" || code === "invalid_request") {
            this.llmPermanentFailures++
            console.error(`[AiScripture] LLM ${String(code)}:`, err?.message || "")
            if (this.llmPermanentFailures >= 2) {
                this.llmStopped = true
                // name the provider & model - a bare code ("model_not_found") is undiagnosable
                const target = this.opts.llm ? ` (${this.opts.llm.provider}: ${this.opts.llm.model || "default model"})` : ""
                this.opts.onStatus("llm_paused", { message: (err?.message || String(code)) + target })
            }
            return
        }

        if (code === "rate_limited") {
            const retryAfter = typeof err?.retryAfter === "number" && Number.isFinite(err.retryAfter) && err.retryAfter > 0 ? err.retryAfter : 15
            this.llmCooldownUntil = Date.now() + Math.min(retryAfter, 60) * 1000
            return
        }

        // timeout/network/server_error/bad_response/refusal: just skip this window & keep going
    }

    // EMISSION

    private tryEmit(candidate: DetectionCandidate, source: "regex" | "llm") {
        const now = Date.now()
        const key = candidate.bookNumber + "." + candidate.chapter
        const keepMs = Math.max(this.cooldownMs, LLM_ALREADY_DETECTED_MS)
        const entries = (this.emitted.get(key) || []).filter((entry) => now - entry.timestamp < keepMs)

        // suppress when it intersects (same book+chapter and overlapping verse range) a reference emitted within the cooldown
        const suppressed = entries.some((entry) => now - entry.timestamp < this.cooldownMs && candidate.verseStart <= entry.verseEnd && candidate.verseEnd >= entry.verseStart)
        if (!suppressed) {
            entries.push({ book: candidate.book, chapter: candidate.chapter, verseStart: candidate.verseStart, verseEnd: candidate.verseEnd, timestamp: now })
            this.opts.onDetection({
                id: "ai-" + now.toString(36) + "-" + (this.idCounter++).toString(36),
                book: candidate.book,
                bookNumber: candidate.bookNumber,
                chapter: candidate.chapter,
                verseStart: candidate.verseStart,
                verseEnd: candidate.verseEnd,
                confidence: candidate.confidence,
                type: candidate.type,
                source,
                quote: candidate.quote,
                spokenBibleId: candidate.spokenBibleId,
                timestamp: now
            })
        }

        this.emitted.set(key, entries)
    }

    private recentDetectionStrings(): string[] {
        const now = Date.now()
        const recent: string[] = []
        this.emitted.forEach((entries) => {
            entries.forEach((entry) => {
                if (now - entry.timestamp < LLM_ALREADY_DETECTED_MS) recent.push(entry.book + " " + entry.chapter + ":" + entry.verseStart + "-" + entry.verseEnd)
            })
        })
        return recent
    }

    // ROLLING TRANSCRIPT

    private trimRollingTranscript() {
        const newestEnd = this.segments[this.segments.length - 1].endMs
        while (this.segments.length > 1 && this.segments[0].endMs < newestEnd - ROLLING_MAX_MS) this.segments.shift()

        let chars = this.segments.reduce((total, segment) => total + segment.text.length, 0)
        while (this.segments.length > 1 && chars > ROLLING_MAX_CHARS) {
            chars -= this.segments[0].text.length
            this.segments.shift()
        }
    }
}

function countWords(text: string): number {
    return text.split(/\s+/).filter((word) => word.length > 0).length
}
