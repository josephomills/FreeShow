// AI AUTO SCRIPTURE - spoken acronym recovery for voice commands
//
// Translation names like NASB are spoken as letters ("en-ay-ess-bee"), and letter runs are
// where streaming ASR is weakest: a live service produced "Give me any be" for "Give me NASB",
// and exact name matching in the command layer can never survive that. Correcting it the way
// asrRepairs.ts corrects book names would mean curating mishearings per acronym per engine;
// instead the acronym itself generates its plausible spoken renderings.
//
// Each all-caps name expands to the concatenated names of its letters ("enayessbee"), plus the
// same with any single letter dropped - a letter lost in transit is exactly the observed failure
// mode ("any be" is "en-ay-bee": NASB minus its S). A candidate phrase is flattened the same way
// and matched by edit distance; only a UNIQUE best acronym within the distance budget counts, so
// two similar acronyms can never steal each other's commands - a miss projects nothing, which on
// a live stage always beats switching the congregation to the wrong translation.

/** How each letter comes out of a speech engine when spoken as a letter name. */
const LETTER_SOUNDS: Record<string, string> = {
    a: "ay",
    b: "be",
    c: "see",
    d: "dee",
    e: "ee",
    f: "ef",
    g: "gee",
    h: "aitch",
    i: "eye",
    j: "jay",
    k: "kay",
    l: "el",
    m: "em",
    n: "en",
    o: "oh",
    p: "pee",
    q: "cue",
    r: "ar",
    s: "es",
    t: "tee",
    u: "you",
    v: "vee",
    w: "double you",
    x: "ex",
    y: "why",
    z: "zee"
}

/** A name the phonetic path applies to: all caps/digits, spoken letter by letter. */
export function isSpokenAcronym(name: string): boolean {
    return /^[A-Z][A-Z0-9]{1,5}$/.test(name.trim())
}

/** Lowercased, spaces and punctuation folded away - the comparison space for spoken letters. */
function flatten(text: string): string {
    return text.toLowerCase().replace(/[^a-z0-9]/g, "")
}

/**
 * The spoken renderings of an acronym: every combination of each letter coming out as its spoken
 * name or as itself (engines do both - "an IV" carries a literal "iv"), for the complete acronym
 * and with each single letter dropped - a letter lost in transit is the observed failure mode.
 */
export function spokenVariants(acronym: string): string[] {
    const letters = acronym.toLowerCase().split("")

    const variants = new Set<string>()
    const addAllRenderings = (subset: string[], minLength: number) => {
        let renderings = [""]
        for (const letter of subset) {
            const sound = LETTER_SOUNDS[letter] || letter
            renderings = renderings.flatMap((prefix) => (sound === letter ? [prefix + letter] : [prefix + sound, prefix + letter]))
        }
        for (const rendering of renderings) {
            const flat = flatten(rendering)
            if (flat.length >= minLength) variants.add(flat)
        }
    }

    addAllRenderings(letters, 0)
    // a dropped letter still has to leave real evidence behind: a short remainder ("ayv" from
    // ASV) sits one edit from ordinary syllables and would switch translations on noise
    if (letters.length >= 3)
        for (let i = 0; i < letters.length; i++)
            addAllRenderings(
                letters.filter((_, index) => index !== i),
                5
            )
    return [...variants]
}

function editDistance(a: string, b: string): number {
    const previous = new Array<number>(b.length + 1)
    for (let j = 0; j <= b.length; j++) previous[j] = j
    for (let i = 1; i <= a.length; i++) {
        let diagonal = previous[0]
        previous[0] = i
        for (let j = 1; j <= b.length; j++) {
            const cost = a[i - 1] === b[j - 1] ? diagonal : diagonal + 1
            diagonal = previous[j]
            previous[j] = Math.min(cost, previous[j] + 1, previous[j - 1] + 1)
        }
    }
    return previous[b.length]
}

export interface AcronymCandidate {
    /** The acronym's owner, opaque here (the caller maps ids). */
    id: string
    acronym: string
}

/**
 * The candidate whose spoken renderings sit closest to the phrase, or null when none is close
 * enough or two are equally close. The budget scales with what was heard: one edit per four
 * characters, so short noise ("me") can never reach any acronym.
 */
export function matchSpokenAcronym(phrase: string, candidates: AcronymCandidate[]): AcronymCandidate | null {
    const heard = flatten(phrase)
    // under four characters there is not enough evidence to tell an acronym from a stray word
    // ("not" is one edit from NET's letters) - and the exact name path already handled short hits
    if (heard.length < 4) return null
    const budget = Math.ceil(heard.length / 4)

    let best: AcronymCandidate | null = null
    let bestDistance = Infinity
    let tied = false

    for (const candidate of candidates) {
        let distance = Infinity
        for (const variant of spokenVariants(candidate.acronym)) distance = Math.min(distance, editDistance(heard, variant))
        if (distance < bestDistance) {
            bestDistance = distance
            best = candidate
            tied = false
        } else if (distance === bestDistance && candidate.acronym !== best?.acronym) tied = true
    }

    if (!best || tied || bestDistance > budget) return null
    return best
}
