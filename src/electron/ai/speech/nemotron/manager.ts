import { app } from "electron"
import fs from "fs"
import path from "path"
import { NEMOTRON_MODEL_FILES, NEMOTRON_VAD_FILE } from "../../setup/models/nemotronFiles"
import { isStampValid, verifyModel, type ModelIntegrity } from "./integrity"

export interface NemotronModelPaths {
    encoder: string
    decoder: string
    joiner: string
    tokens: string
}

// file names are shared with the setup layer (setup/models/nemotron.ts), so the downloader
// and this runtime loader can never disagree about where the model lives
const MODEL_FILE_NAMES = [...Object.values(NEMOTRON_MODEL_FILES).map((entry) => entry.file), NEMOTRON_VAD_FILE]

const MIN_FILE_SIZE = 1024 // anything smaller is a broken download or an error page

// PATHS

export function getModelDir(): string {
    const dir = path.join(app.getPath("userData"), "bin", "nemotron", "models")

    // one-time move of a download made before the bin/<engine>/models convention
    const legacy = path.join(app.getPath("userData"), "nemotron-model")
    if (!fs.existsSync(dir) && fs.existsSync(legacy)) {
        try {
            fs.mkdirSync(path.dirname(dir), { recursive: true })
            fs.renameSync(legacy, dir)
        } catch (err) {
            console.error("[nemotron] Could not move the model to its new location:", err)
            return legacy
        }
    }

    // one-time move of files an earlier downloader placed in bin/nemotron instead of bin/nemotron/models
    for (const file of MODEL_FILE_NAMES) {
        const stray = path.join(path.dirname(dir), file)
        try {
            if (!fs.existsSync(stray) || fs.existsSync(path.join(dir, file))) continue
            fs.mkdirSync(dir, { recursive: true })
            fs.renameSync(stray, path.join(dir, file))
        } catch (err) {
            console.error("[nemotron] Could not move a model file to its new location:", err)
        }
    }

    return dir
}

export function getNemotronModelPaths(): NemotronModelPaths | null {
    const dir = getModelDir()
    const paths: NemotronModelPaths = {
        encoder: path.join(dir, NEMOTRON_MODEL_FILES.encoder.file),
        decoder: path.join(dir, NEMOTRON_MODEL_FILES.decoder.file),
        joiner: path.join(dir, NEMOTRON_MODEL_FILES.joiner.file),
        tokens: path.join(dir, NEMOTRON_MODEL_FILES.tokens.file)
    }

    for (const file of Object.values(paths)) {
        if (!isUsableFile(file)) return null
    }
    return paths
}

export function getVadModelPath(): string | null {
    const file = path.join(getModelDir(), NEMOTRON_VAD_FILE)
    return isUsableFile(file) ? file : null
}

/**
 * Whether the model on disk is the one this build pins. Hashes on the first call after a download
 * or an upgrade (~1.7s for 662 MB) and is instant afterwards - see integrity.ts.
 *
 * Kept separate from getNemotronModelPaths() because they answer different questions: that one asks
 * whether anything is there, this one asks whether it is the right thing. Loading a superseded
 * model is silent - it still transcribes, just not as the code was measured against.
 */
export async function getNemotronModelIntegrity(): Promise<ModelIntegrity> {
    return verifyModel(getModelDir())
}

/** The cheap form: true only when a previous verification still holds. Never hashes. */
export function isNemotronModelVerified(): boolean {
    return isStampValid(getModelDir())
}

/** The native addon is optional - report whether this platform can actually run it. */
export function isNemotronSupported(): boolean {
    try {
        require.resolve("sherpa-onnx-node")
        return true
    } catch {
        return false
    }
}

function isUsableFile(file: string): boolean {
    try {
        return fs.existsSync(file) && fs.statSync(file).size > MIN_FILE_SIZE
    } catch {
        return false
    }
}
