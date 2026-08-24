// AI AUTO SCRIPTURE - non-speech labels the engines emit, shared by both.
//
// Every engine here labels audio it judges not to be speech rather than inventing words for it:
// "[MUSIC PLAYING]", "(upbeat music)", "[BLANK_AUDIO]", "*applause*", "♪". whisper/transcriber.ts
// has recognised these since it was written; the streaming engine did not, so its labels reached
// the transcript as ordinary speech and scripture detection treated them as something a preacher
// had said.
//
// A label is worth showing - it tells an operator the room is between things - but it must never
// feed detection, and neither must sung content, because an ASR asked to transcribe a song it does
// not know will produce confident nonsense.

/** Nothing but bracketed, parenthesised or starred labels and punctuation. */
export function isAnnotationOnly(text: string): boolean {
    const leftover = text.replace(/\[[^\]]*\]|\([^)]*\)|\*[^*]*\*/g, "").replace(/[♪♫\s.,!?\-–—_]+/g, "")
    return leftover === ""
}

const MUSIC_WORDS = /\b(music|musical|singing|sung|song|songs|instrumental|humming|chanting|applause|cheering)\b/i

/**
 * Words describing music rather than words sung to it.
 *
 * whisper marks sung content by wrapping it in ♪. The streaming models label it in prose instead -
 * "[MUSIC PLAYING]", "(upbeat music)", "(singing)" - so both shapes count.
 */
export function isMusicAnnotation(text: string): boolean {
    if (/[♪♫]/.test(text)) return true

    // A label can arrive split across segments - the streaming driver commits whole words, so
    // "[MUSIC PLAYING]" may reach here as "[MUSIC" and then "PLAYING]". The closing bracket is
    // therefore optional at the END of the text, where an unfinished label is what that looks like,
    // but required anywhere else so a stray bracket mid-sentence cannot swallow the rest of it.
    const labels = text.match(/\[[^\]]*\]|\([^)]*\)|\*[^*]*\*|[[(*][^\])*]*$/g)
    if (!labels) return false
    return labels.some((label) => MUSIC_WORDS.test(label))
}
