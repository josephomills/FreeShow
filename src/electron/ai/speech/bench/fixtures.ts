// AI BENCH - fixture manifests.
//
// No audio is committed: 16 kHz mono s16 is ~1.9 MB per minute, and the fixtures that matter most
// (real services) cannot be committed at all for privacy and copyright reasons. What lives in the
// repo is the MANIFEST - the reference transcript and the expected scripture references with the
// audio-time at which each spoken phrase ends. Audio is generated locally (synthetic tier),
// downloaded and hash-pinned (public-domain tier), or pointed at by the operator (private tier).
//
// `phraseEndMs` is the anchor every latency number hangs off, so how it was obtained is recorded
// per fixture rather than assumed: an exact value from a generator is a different kind of number
// from a hand-marked one, and a comparison that mixes them should say so.

import fs from "fs"
import os from "os"
import path from "path"
import { createHash } from "crypto"

export type FixtureTier =
    /** Generated locally by scripts/ai/makeFixtures.js. Timings exact by construction. */
    | "synthetic"
    /** synthetic, then degraded (room impulse response + babble at a target SNR). */
    | "degraded"
    /** Public-domain recordings, downloaded and SHA-pinned. Timings hand-marked. */
    | "public"
    /** Real preaching - the primary use case. Operator-supplied, never committed. */
    | "sermon"

export type TimingSource = "generated" | "hand-marked" | "none"

/**
 * One entry per reference that SHOULD REACH THE SCREEN - not one per time the words were spoken.
 *
 * The coordinator suppresses an intersecting reference emitted within refCooldownSeconds (90 by
 * default), so a preacher repeating "Matthew 6:33" three times in twenty seconds should produce a
 * single detection. Listing all three scored that correct behaviour as two misses, which is how a
 * recall number ends up arguing against the feature working properly.
 */
export interface ExpectedReference {
    /** 1-based book number, matching the detection layer's canon numbering. */
    book: number
    chapter: number
    verseStart: number
    verseEnd?: number
    /** The words actually spoken, e.g. "Ephesians chapter two verse eight". */
    phrase: string
    /** Audio-time (ms) at which the last word of `phrase` finishes. The latency anchor. */
    phraseEndMs: number
}

export interface Fixture {
    id: string
    tier: FixtureTier
    /** Relative to the fixture root. */
    file: string
    /** Absolute, filled in by loadFixtures(). */
    absolutePath: string
    durationMs?: number
    /** Ground-truth words. Absent for detection-only fixtures. */
    transcript?: string
    timingSource: TimingSource
    expected: ExpectedReference[]
    /** Signal-to-noise ratio in dB for the degraded tier. */
    snrDb?: number
    notes?: string
}

export interface FixtureSet {
    id: string
    description: string
    fixtures: Fixture[]
    /** Stable across machines: manifest content, not file bytes. Guards cross-set comparisons. */
    hash: string
}

/** Where generated and downloaded fixture audio lives. Never inside the repo. */
export function resolveFixtureRoot(): string {
    if (process.env.FREESHOW_AI_FIXTURES) return process.env.FREESHOW_AI_FIXTURES

    const home = os.homedir()
    const base = process.platform === "darwin" ? path.join(home, "Library", "Application Support", "FreeShow") : process.platform === "win32" ? path.join(process.env.APPDATA || path.join(home, "AppData", "Roaming"), "FreeShow") : path.join(process.env.XDG_CONFIG_HOME || path.join(home, ".config"), "FreeShow")

    return path.join(base, "bin", "bench", "fixtures")
}

/** Committed manifests live next to the code; generated audio does not. */
export function resolveManifestDir(): string {
    return path.join(__dirname, "manifests")
}

interface RawManifest {
    id: string
    description: string
    fixtures: Omit<Fixture, "absolutePath">[]
}

function hashManifest(raw: RawManifest): string {
    // hash the manifest, not the audio: a regenerated synthetic clip is byte-different but
    // semantically identical, and a report should not refuse to compare across that
    return createHash("sha256").update(JSON.stringify(raw)).digest("hex").slice(0, 12)
}

export function loadFixtureSet(manifestPath: string): FixtureSet {
    const raw = JSON.parse(fs.readFileSync(manifestPath, "utf8")) as RawManifest
    const root = resolveFixtureRoot()

    return {
        id: raw.id,
        description: raw.description,
        hash: hashManifest(raw),
        fixtures: raw.fixtures.map((fixture) => ({
            ...fixture,
            absolutePath: path.isAbsolute(fixture.file) ? fixture.file : path.join(root, fixture.file)
        }))
    }
}

/** Fixtures whose audio is actually on disk. Missing audio is skipped, never an error. */
export function availableFixtures(set: FixtureSet): Fixture[] {
    return set.fixtures.filter((fixture) => {
        try {
            return fs.existsSync(fixture.absolutePath) && fs.statSync(fixture.absolutePath).size > 44
        } catch {
            return false
        }
    })
}

/**
 * Manifests come from two places, and the split is a privacy boundary rather than a convenience.
 * Committed manifests (bench/manifests/) describe audio anyone can obtain. Manifests found in the
 * fixture root describe recordings that belong to whoever ran the bench - real services, private
 * sermon archives - and neither the audio NOR the manifest may enter the repo, because the file
 * names alone identify the source. makeFixtures.js therefore writes its manifest beside the audio.
 */
export function listManifests(): string[] {
    const dirs = [resolveManifestDir(), resolveFixtureRoot()]
    const found: string[] = []

    for (const dir of dirs) {
        try {
            for (const file of fs.readdirSync(dir)) {
                if (file.endsWith(".json")) found.push(path.join(dir, file))
            }
        } catch {
            // a missing directory just means that tier is not set up here
        }
    }
    return found
}
