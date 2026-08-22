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
}

export function createDriver(variant: EngineVariant, callbacks: DriverCallbacks, language = "en"): TranscriptionDriver {
    // both drivers are constructed here rather than in the runner, so every metric path is
    // identical between them and the only thing that differs is the decode strategy
    if (variant.engine !== "nemotron") throw new Error(`Unknown bench engine: ${variant.engine}`)

    const paths = findNemotronPaths()
    if (!paths) throw new Error(`Nemotron model not found in ${resolveNemotronModelDir()}`)

    const options = {
        paths: { encoder: paths.encoder, decoder: paths.decoder, joiner: paths.joiner, tokens: paths.tokens },
        vadModelPath: paths.vad,
        language,
        ...callbacks
    }
    return variant.decode === "stream" ? new NemotronStreamDriver(options) : new NemotronDriver(options)
}

/** The A/B pair every phase-2 claim is measured against. */
export const VARIANTS: EngineVariant[] = [
    { id: "nemotron/batch", engine: "nemotron", decode: "batch" },
    { id: "nemotron/stream", engine: "nemotron", decode: "stream" }
]
