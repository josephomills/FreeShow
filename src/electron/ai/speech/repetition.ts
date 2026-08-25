// AI AUTO SCRIPTURE - phrase-level repetition, the streaming decoder's degenerate failure.
//
// A greedy RNN-T can lock into a cycle and emit the same phrase indefinitely: "and he saith the
// LORD, and he saith the LORD, ..." was seen filling a live transcript. The predictor's own output
// is its next input, so once it enters the cycle nothing in the audio pulls it out - only clearing
// the decoder state does.
//
// whisper/transcriber.ts already has isRepetitionLoop for its own version of this, but that test is
// whether ONE token carries most of a segment ("heh, heh, heh"). A repeating PHRASE defeats it: in
// "and he saith the LORD" five times over, no single word is more than a quarter of the text.

/** Longest phrase considered - beyond this a repeat is more likely to be real speech. */
const MAX_PHRASE_TOKENS = 8

/**
 * How many times a phrase must repeat back-to-back before it is a decoder loop rather than a person
 * speaking, by phrase length.
 *
 * Preaching repeats deliberately - a word said three or four times for emphasis is a rhetorical
 * device, not a fault, and treating it as one clears the decoder mid-sentence and corrupts what
 * follows. The observed decoder loops ran to twenty repeats and more, so there is a lot of room
 * between the two.
 *
 * A single word tolerates the most, because that is what a speaker actually does. Repeating a
 * four-word phrase even four times over is not something people say.
 */
function minRepeatsFor(phraseTokens: number): number {
    if (phraseTokens === 1) return 8
    if (phraseTokens === 2) return 6
    return 4
}

/** The smallest threshold any phrase length uses - below this nothing can qualify. */
const MIN_REPEATS = 4

interface Token {
    text: string
    /** Character offset in the source string, so a caller can cut without re-joining. */
    at: number
}

function tokenize(text: string): Token[] {
    const tokens: Token[] = []
    const pattern = /\S+/g
    let match: RegExpExecArray | null
    while ((match = pattern.exec(text)) !== null) tokens.push({ text: match[0].toLowerCase().replace(/[^\p{L}\p{N}']/gu, ""), at: match.index })
    return tokens.filter((token) => token.text)
}

/**
 * Where a repeating tail begins, or -1.
 *
 * Only the TAIL is considered, because that is where a live decoder is looping right now - a phrase
 * repeated earlier and then left behind is something the speaker said. Returns a character offset
 * into `text` so the caller can keep the first occurrence and discard the cycle.
 */
export function findRepeatedTail(text: string): number {
    const tokens = tokenize(text)
    if (tokens.length < MIN_REPEATS * 2) return -1

    for (let size = 1; size <= MAX_PHRASE_TOKENS; size++) {
        const needed = minRepeatsFor(size)
        if (tokens.length < size * needed) continue

        const phrase = tokens.slice(tokens.length - size).map((token) => token.text)
        let repeats = 1
        let start = tokens.length - size

        while (start - size >= 0) {
            const previous = tokens.slice(start - size, start).map((token) => token.text)
            if (previous.join(" ") !== phrase.join(" ")) break
            start -= size
            repeats++
        }

        if (repeats >= needed) return tokens[start + size].at
    }
    return -1
}

/** True when the tail of `text` is a decoder loop rather than speech. */
export function hasRepeatedTail(text: string): boolean {
    return findRepeatedTail(text) >= 0
}

export interface RepetitionSummary {
    /** Words inside a phrase repeated back to back at least MIN_REPEATS times. */
    loopedWords: number
    totalWords: number
    /** Distinct runs, so one long cycle is not confused with many short ones. */
    runs: number
    /** The longest run, in words - what a viewer would actually see fill the screen. */
    longestRun: number
    share: number
}

/**
 * Scan a whole transcript for cycles, not just its tail.
 *
 * The tail check is what a live driver needs; this is what a BENCHMARK needs, because degeneration
 * is only visible over minutes of continuous decoding and a fixture set of two-minute clips cannot
 * show it at all. Reporting the share of a transcript lost to cycles is the metric that would have
 * caught a regression measured as an improvement.
 */
export function summarizeRepetition(text: string): RepetitionSummary {
    const tokens = tokenize(text)
    const looped = new Array<boolean>(tokens.length).fill(false)
    let runs = 0
    let longestRun = 0

    for (let start = 0; start < tokens.length; start++) {
        if (looped[start]) continue

        for (let size = 1; size <= MAX_PHRASE_TOKENS; size++) {
            const needed = minRepeatsFor(size)
            if (start + size * needed > tokens.length) continue

            const phrase = tokens
                .slice(start, start + size)
                .map((token) => token.text)
                .join(" ")
            let repeats = 1
            while (start + size * (repeats + 1) <= tokens.length) {
                const next = tokens
                    .slice(start + size * repeats, start + size * (repeats + 1))
                    .map((token) => token.text)
                    .join(" ")
                if (next !== phrase) break
                repeats++
            }

            if (repeats >= needed) {
                const length = size * repeats
                for (let at = start; at < start + length; at++) looped[at] = true
                runs++
                longestRun = Math.max(longestRun, length)
                start += length - 1
                break
            }
        }
    }

    const loopedWords = looped.filter(Boolean).length
    return { loopedWords, totalWords: tokens.length, runs, longestRun, share: tokens.length ? loopedWords / tokens.length : 0 }
}
