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
/** How many times a phrase must repeat back-to-back before it is a loop rather than emphasis. */
const MIN_REPEATS = 3

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
        if (tokens.length < size * MIN_REPEATS) break

        const phrase = tokens.slice(tokens.length - size).map((token) => token.text)
        let repeats = 1
        let start = tokens.length - size

        while (start - size >= 0) {
            const previous = tokens.slice(start - size, start).map((token) => token.text)
            if (previous.join(" ") !== phrase.join(" ")) break
            start -= size
            repeats++
        }

        // a single word repeated twice is emphasis ("very, very"); a phrase repeated three times is
        // not something a person says
        if (repeats >= MIN_REPEATS) return tokens[start + size].at
    }
    return -1
}

/** True when the tail of `text` is a decoder loop rather than speech. */
export function hasRepeatedTail(text: string): boolean {
    return findRepeatedTail(text) >= 0
}
