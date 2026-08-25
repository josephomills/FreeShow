// AI AUTO SCRIPTURE - SHARED SESSION STATE
// the mutable state the scripture modules genuinely share (ES module bindings are not writable
// across modules, so the cross-module fields live on one object). State a single module owns -
// timers, debounce tokens, status-filter memory - stays private to that module instead.

import { get } from "svelte/store"
import type { DetectedReference } from "../../../types/ai/AiScripture"
import type { OutSlide } from "../../../types/Show"
import { ai } from "../../stores"

export type PreviousOutputState = {
    activeScripture: { id?: string; reference?: { book: number | string; chapters: (number | string)[]; verses: (number | string)[][] } }
    outSlide: OutSlide | null
}

export const scriptureState = {
    sessionActive: false,
    // the searched local translations in priority order - session bibles / quote verification / spoken cycling all read it
    searchBibleIds: [] as string[],
    // marks our own projections so the manual-override watcher never mistakes them for operator actions
    selfProjecting: false,
    // how many spoken "go back" steps have walked the scripture history
    backDepth: 0,
    // snapshot of the output before the AI projected, for "bring it back"
    previousState: null as PreviousOutputState | null,
    lastAutoProjectionAt: 0,
    lastAutoProjectedRef: null as DetectedReference | null,
    lastAutoProjectedBibleId: "", // which translation's wording is on the output right now
    lastQuoteMatchAnchor: null as { bookNumber: number; chapter: number; verseStart: number; verseEnd: number } | null,
    // the drawer translation as the session STARTED - the last-resort projection target when no
    // main translation and no favourites are configured. The live drawer tab must never be that
    // fallback: our own projection follow moves the drawer, so one matched-mode projection in
    // another translation would drag every later spoken reference to it (seen live: a single WEB
    // quote match turned a whole reading's "verse N" projections into WEB, uncalled for)
    sessionFallbackTranslationId: ""
}

export function getSettings() {
    return get(ai).scripture || {}
}
