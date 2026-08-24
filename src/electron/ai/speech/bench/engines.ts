// AI BENCH - construct a real TranscriptionDriver outside electron.
//
// The bench runs under vitest/node, where `electron` cannot be imported. NemotronDriver itself is
// clean (it only pulls ../seam and the shared types, and lazy-requires the native addon), so the
// only thing standing between the bench and a real decode is that speech/nemotron/manager.ts
// resolves the model directory through app.getPath("userData"). This file does that resolution
// directly, using the same file names via setup/models/nemotronFiles.ts - so a renamed model file
// can never make the bench and the app disagree about what they are measuring.

import fs from "fs"
import os from "os"
import path from "path"
import { NEMOTRON_MODEL_FILES, NEMOTRON_VAD_FILE } from "../../setup/models/nemotronFiles"
import { NemotronDriver } from "../nemotron/driver"
import { NemotronStreamDriver } from "../nemotron/streamDriver"
import { findModelSet } from "./modelSets"
import type { DriverCallbacks, TranscriptionDriver } from "../types"

export type BenchEngineId = "nemotron"

/** Electron's app.getPath("userData") for FreeShow, without electron. */
export function resolveUserDataDir(): string {
    const override = process.env.FREESHOW_USER_DATA
    if (override) return override

    const home = os.homedir()
    if (process.platform === "darwin") return path.join(home, "Library", "Application Support", "FreeShow")
    if (process.platform === "win32") return path.join(process.env.APPDATA || path.join(home, "AppData", "Roaming"), "FreeShow")
    return path.join(process.env.XDG_CONFIG_HOME || path.join(home, ".config"), "FreeShow")
}

/** Where the app downloads the streaming model. FREESHOW_AI_MODEL_DIR points at a different set. */
export function resolveNemotronModelDir(): string {
    return process.env.FREESHOW_AI_MODEL_DIR || path.join(resolveUserDataDir(), "bin", "nemotron", "models")
}

/** Alternative model sets, downloaded by scripts/ai/getBenchModels.js. Never shipped. */
export function resolveBenchModelDir(setId: string): string {
    return path.join(resolveUserDataDir(), "bin", "bench", "models", setId)
}

/** The VAD is shared by every set - only the app downloads it, and it is model-independent. */
function resolveVadPath(): string | null {
    const file = path.join(resolveNemotronModelDir(), NEMOTRON_VAD_FILE)
    return isUsable(file) ? file : null
}

function isUsable(file: string): boolean {
    try {
        return fs.existsSync(file) && fs.statSync(file).size > 1024
    } catch {
        return false
    }
}

/** Model files for a bench set, or null when it has not been downloaded. */
export function findBenchModelPaths(setId: string): NemotronPaths | null {
    const dir = resolveBenchModelDir(setId)
    const vad = resolveVadPath()
    if (!vad) return null

    const paths: NemotronPaths = {
        encoder: path.join(dir, "encoder.int8.onnx"),
        decoder: path.join(dir, "decoder.int8.onnx"),
        joiner: path.join(dir, "joiner.int8.onnx"),
        tokens: path.join(dir, "tokens.txt"),
        vad
    }
    return Object.values(paths).every(isUsable) ? paths : null
}

export function benchModelReady(setId?: string): boolean {
    if (!setId) return !!findNemotronPaths()
    return !!findBenchModelPaths(setId)
}

export interface NemotronPaths {
    encoder: string
    decoder: string
    joiner: string
    tokens: string
    vad: string
}

/** null when the model is not downloaded - the caller skips rather than fails. */
export function findNemotronPaths(): NemotronPaths | null {
    const dir = resolveNemotronModelDir()
    const paths: NemotronPaths = {
        encoder: path.join(dir, NEMOTRON_MODEL_FILES.encoder.file),
        decoder: path.join(dir, NEMOTRON_MODEL_FILES.decoder.file),
        joiner: path.join(dir, NEMOTRON_MODEL_FILES.joiner.file),
        tokens: path.join(dir, NEMOTRON_MODEL_FILES.tokens.file),
        vad: path.join(dir, NEMOTRON_VAD_FILE)
    }

    for (const file of Object.values(paths)) {
        try {
            if (!fs.existsSync(file) || fs.statSync(file).size <= 1024) return null
        } catch {
            return null
        }
    }
    return paths
}

export function hasSherpa(): boolean {
    try {
        require.resolve("sherpa-onnx-node")
        return true
    } catch {
        return false
    }
}

/** Everything a bench run needs before it is worth starting. */
export function benchEngineReady(engine: BenchEngineId): boolean {
    if (engine === "nemotron") return hasSherpa() && !!findNemotronPaths()
    return false
}

export interface EngineVariant {
    /** Stamped into the report so two runs are never compared across different engine builds. */
    id: string
    engine: BenchEngineId
    /** "batch" is the shipped fresh-stream-per-decode path; "stream" keeps one warm stream. */
    decode: "batch" | "stream"
    /** A set id from modelSets.ts. Omitted means the model the app itself downloaded. */
    modelSet?: string
    /** modelConfig.language for a multilingual export. */
    modelLanguage?: string
    /** Merged into the recognizer config - decodingMethod, hotwordsFile, bpeVocab, ... */
    recognizerOverrides?: Record<string, unknown>
    /** Minimum audio between decoder resets; 0 resets at every utterance boundary. */
    resetIntervalMs?: number
}

export function createDriver(variant: EngineVariant, callbacks: DriverCallbacks, language = "en"): TranscriptionDriver {
    // both drivers are constructed here rather than in the runner, so every metric path is
    // identical between them and the only thing that differs is the decode strategy
    if (variant.engine !== "nemotron") throw new Error(`Unknown bench engine: ${variant.engine}`)

    const paths = variant.modelSet ? findBenchModelPaths(variant.modelSet) : findNemotronPaths()
    if (!paths) throw new Error(`model set ${variant.modelSet || "(app)"} not found - run: node scripts/ai/getBenchModels.js ${variant.modelSet || ""}`)

    const options = {
        paths: { encoder: paths.encoder, decoder: paths.decoder, joiner: paths.joiner, tokens: paths.tokens },
        vadModelPath: paths.vad,
        language,
        modelLanguage: variant.modelLanguage,
        // every commit timing scales off the export's grid, so a 160ms set benched with the
        // shipped 1120ms constants would look far slower than it is
        chunkShiftMs: variant.modelSet ? findModelSet(variant.modelSet)?.chunkMs : undefined,
        featureDim: variant.modelSet ? findModelSet(variant.modelSet)?.featureDim : undefined,
        resetIntervalMs: variant.resetIntervalMs,
        recognizerOverrides: variant.recognizerOverrides,
        ...callbacks
    }
    // only the streaming driver can carry an alternative model set: the batch driver is the
    // baseline being compared against, and there is no reason to vary two things at once
    return variant.decode === "stream" ? new NemotronStreamDriver(options) : new NemotronDriver(options)
}

/**
 * The comparison matrix. Ordered so each row changes ONE thing from the row above it, which is
 * what makes the deltas attributable:
 *
 *   batch -> stream       how the recognizer is driven   (same model, same tier)
 *   en-1120 -> en-160     the chunk tier                 (same weights)
 *   en-1120 -> multi-1120 English-only -> multilingual   (same tier)
 *   multi-1120 -> multi-320  the tier again, on the multilingual weights
 *
 * Rows whose model set is not downloaded are skipped rather than failing.
 */
export const VARIANTS: EngineVariant[] = [
    { id: "batch en-1120", engine: "nemotron", decode: "batch" },
    { id: "stream en-1120", engine: "nemotron", decode: "stream" },
    { id: "stream en-160", engine: "nemotron", decode: "stream", modelSet: "en-160" },
    { id: "stream multi-1120", engine: "nemotron", decode: "stream", modelSet: "multi-1120", modelLanguage: "en" },
    { id: "stream multi-320", engine: "nemotron", decode: "stream", modelSet: "multi-320", modelLanguage: "en" },
    { id: "multi-1120 warm", engine: "nemotron", decode: "stream", modelSet: "multi-1120", modelLanguage: "en", resetIntervalMs: 600_000 },
    { id: "zipformer", engine: "nemotron", decode: "stream", modelSet: "zipformer-en" },
    { id: "zipformer-multi", engine: "nemotron", decode: "stream", modelSet: "zipformer-multi" }
]

/** Variants whose model set is present on this machine. */
export function availableVariants(variants: EngineVariant[] = VARIANTS): EngineVariant[] {
    return variants.filter((variant) => benchModelReady(variant.modelSet))
}
