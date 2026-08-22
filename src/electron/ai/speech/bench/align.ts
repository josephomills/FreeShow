// AI BENCH - reference/hypothesis alignment.
//
// Word error rate needs more than an edit distance: every downstream metric wants to know WHICH
// hypothesis word a reference word became, so a reference word's emission time can be looked up.
// So this keeps the backtrace, not just the score.
//
// On normalization: this is the WER convention (case, punctuation and British/American spelling
// folded away), NOT the quote matcher's. quoteMatchTokens.ts deliberately drops single-character
// tokens and folds apostrophes because it is matching damaged speech against verse text; doing
// that here would silently forgive real deletions and flatter every engine. Word BOUNDARIES are
// defined the same way in both, so a token is the same thing on either side.

/** Contractions the engines expand or contract inconsistently - neither spelling is an error. */
const CONTRACTIONS: Record<string, string> = {
    cant: "cannot",
    dont: "do not",
    doesnt: "does not",
    didnt: "did not",
    wont: "will not",
    wouldnt: "would not",
    couldnt: "could not",
    shouldnt: "should not",
    isnt: "is not",
    arent: "are not",
    wasnt: "was not",
    werent: "were not",
    hasnt: "has not",
    havent: "have not",
    hadnt: "had not",
    its: "it is",
    thats: "that is",
    theres: "there is",
    hes: "he is",
    shes: "she is",
    theyre: "they are",
    were: "we are",
    youre: "you are",
    im: "i am",
    ive: "i have",
    ill: "i will",
    lets: "let us"
}

/**
 * British -> American, applied to both sides. LibriVox and KJV-adjacent references are full of
 * these ("dishonoured", "saviour", "honour"), and scoring them as errors would say more about the
 * corpus than the engine. Suffix rules rather than a word list, so it generalizes.
 */
function foldSpelling(token: string): string {
    return token
        .replace(/([a-z])our(s?)$/, "$1or$2") // honour -> honor, colours -> colors
        .replace(/([a-z])oured$/, "$1ored")
        .replace(/([a-z])ouring$/, "$1oring")
        .replace(/ise$/, "ize")
        .replace(/ised$/, "ized")
        .replace(/ising$/, "izing")
        .replace(/isation$/, "ization")
        .replace(/([a-z])re$/, (match, prev) => (/[bcdfgklmnpqstvxz]/.test(prev) ? `${prev}er` : match)) // centre -> center
}

/**
 * Word boundaries match quoteMatchTokens.ts's baseTokens: lowercase, strip html and diacritics,
 * keep letters/numbers/apostrophes, fold apostrophes away. Unlike that function, nothing is
 * dropped for being short - "a" and "I" are words and deleting one is an error.
 */
export function normalizeForWer(text: string): string[] {
    return text
        .toLowerCase()
        .normalize("NFD")
        .replace(/\p{Diacritic}/gu, "")
        .replace(/<[^>]*>/g, " ")
        .replace(/[^\p{L}\p{N}\s']/gu, " ")
        .replace(/'/g, "")
        .split(/\s+/)
        .filter(Boolean)
        .flatMap((token) => (CONTRACTIONS[token] ? CONTRACTIONS[token].split(" ") : [token]))
        .map(foldSpelling)
}

export type EditOp = "match" | "substitute" | "delete" | "insert"

export interface AlignedPair {
    op: EditOp
    /** Index into the reference tokens, or -1 for an insertion. */
    refIndex: number
    /** Index into the hypothesis tokens, or -1 for a deletion. */
    hypIndex: number
    refToken?: string
    hypToken?: string
}

export interface Alignment {
    pairs: AlignedPair[]
    substitutions: number
    deletions: number
    insertions: number
    hits: number
    refLength: number
    /** (S + D + I) / N. Can exceed 1 when the hypothesis is much longer than the reference. */
    wer: number
}

/** Levenshtein over tokens, keeping the backtrace. */
export function align(reference: string[], hypothesis: string[]): Alignment {
    const n = reference.length
    const m = hypothesis.length

    // (n+1) x (m+1) cost table. Fixtures are minutes, not hours, so the full table is fine.
    const cost: number[][] = Array.from({ length: n + 1 }, () => new Array<number>(m + 1).fill(0))
    for (let i = 0; i <= n; i++) cost[i][0] = i
    for (let j = 0; j <= m; j++) cost[0][j] = j

    for (let i = 1; i <= n; i++) {
        for (let j = 1; j <= m; j++) {
            const substitute = cost[i - 1][j - 1] + (reference[i - 1] === hypothesis[j - 1] ? 0 : 1)
            cost[i][j] = Math.min(substitute, cost[i - 1][j] + 1, cost[i][j - 1] + 1)
        }
    }

    const pairs: AlignedPair[] = []
    let substitutions = 0
    let deletions = 0
    let insertions = 0
    let hits = 0

    let i = n
    let j = m
    while (i > 0 || j > 0) {
        // prefer the diagonal on ties: a substitution carries more information for the caller
        // than a delete+insert pair at the same cost, and keeps refIndex/hypIndex paired up
        if (i > 0 && j > 0 && cost[i][j] === cost[i - 1][j - 1] + (reference[i - 1] === hypothesis[j - 1] ? 0 : 1)) {
            const isMatch = reference[i - 1] === hypothesis[j - 1]
            pairs.push({ op: isMatch ? "match" : "substitute", refIndex: i - 1, hypIndex: j - 1, refToken: reference[i - 1], hypToken: hypothesis[j - 1] })
            if (isMatch) hits++
            else substitutions++
            i--
            j--
        } else if (i > 0 && cost[i][j] === cost[i - 1][j] + 1) {
            pairs.push({ op: "delete", refIndex: i - 1, hypIndex: -1, refToken: reference[i - 1] })
            deletions++
            i--
        } else {
            pairs.push({ op: "insert", refIndex: -1, hypIndex: j - 1, hypToken: hypothesis[j - 1] })
            insertions++
            j--
        }
    }
    pairs.reverse()

    return {
        pairs,
        substitutions,
        deletions,
        insertions,
        hits,
        refLength: n,
        wer: n === 0 ? (m === 0 ? 0 : 1) : (substitutions + deletions + insertions) / n
    }
}

/**
 * WER restricted to reference words in `vocabulary` - the number that prices contextual biasing.
 * Overall WER is dominated by function words the engine always gets right, so a decoder that
 * mangles every biblical proper noun can still post a respectable headline figure.
 */
export function vocabularyErrorRate(alignment: Alignment, vocabulary: Set<string>): { errors: number; total: number; rate: number; missed: string[] } {
    let errors = 0
    let total = 0
    const missed: string[] = []

    for (const pair of alignment.pairs) {
        if (pair.refIndex < 0 || !pair.refToken || !vocabulary.has(pair.refToken)) continue
        total++
        if (pair.op !== "match") {
            errors++
            missed.push(pair.refToken)
        }
    }

    return { errors, total, rate: total === 0 ? 0 : errors / total, missed }
}
