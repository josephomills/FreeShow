// AI AUTO SCRIPTURE - is the model on disk the model we pinned?
//
// The downloader verifies every file's SHA-256 as it lands, but nothing checked afterwards: the
// runtime loader only asked whether the files existed and were bigger than a kilobyte. So a model
// from an older pinned revision kept being used indefinitely, and the app reported itself ready -
// the failure is invisible, because a superseded model still transcribes. It just transcribes
// worse, or differently, than the one the code was written and measured against.
//
// Hashing 662 MB costs about 1.7s, which is too much to pay on every session start, so a
// successful verification is stamped beside the model and the stamp is trusted while the files
// are byte-identical to what was hashed. The stamp records size and mtime as well as the digest,
// so a file swapped underneath it fails the cheap check and forces a rehash rather than being
// trusted on the strength of its name.

import { createHash } from "crypto"
import fs from "fs"
import path from "path"
import { pipeline } from "stream/promises"
import { NEMOTRON_MODEL_FILES, NEMOTRON_MODEL_REVISION } from "../../setup/models/nemotronFiles"

const STAMP_FILE = "model.json"

interface StampedFile {
    sha256: string
    size: number
    mtimeMs: number
}

interface ModelStamp {
    /** The pinned repo revision these hashes came from. */
    revision: string
    verifiedAtMs: number
    files: { [name: string]: StampedFile }
}

export type ModelIntegrity =
    /** Every pinned file is present and hashes to the pinned digest. */
    | "ok"
    /** A pinned file is absent or unreadable - the model was never downloaded, or was deleted. */
    | "missing"
    /** Every file is present but at least one is not the pinned bytes - an older revision. */
    | "outdated"

export async function sha256File(filePath: string): Promise<string> {
    const hash = createHash("sha256")
    await pipeline(fs.createReadStream(filePath), hash)
    return hash.digest("hex")
}

function stampPath(modelDir: string): string {
    return path.join(modelDir, STAMP_FILE)
}

function statOf(file: string): fs.Stats | null {
    try {
        return fs.statSync(file)
    } catch {
        return null
    }
}

function readStamp(modelDir: string): ModelStamp | null {
    try {
        const stamp = JSON.parse(fs.readFileSync(stampPath(modelDir), "utf8")) as ModelStamp
        return stamp?.files && stamp?.revision ? stamp : null
    } catch {
        return null
    }
}

/** Record what was verified, so the next start does not re-hash 662 MB. Failure is not fatal. */
function writeStamp(modelDir: string, files: { [name: string]: StampedFile }) {
    try {
        const stamp: ModelStamp = { revision: NEMOTRON_MODEL_REVISION, verifiedAtMs: Date.now(), files }
        fs.writeFileSync(stampPath(modelDir), JSON.stringify(stamp, null, 4))
    } catch (err) {
        // a read-only or full disk costs a re-hash next time, nothing worse
        console.error("[nemotron] Could not record the model verification stamp:", err)
    }
}

/**
 * Cheap check: a stamp from this revision, whose recorded digests are the pinned ones, for files
 * that have not changed size or mtime since. Never hashes.
 */
export function isStampValid(modelDir: string): boolean {
    const stamp = readStamp(modelDir)
    if (!stamp || stamp.revision !== NEMOTRON_MODEL_REVISION) return false

    for (const entry of Object.values(NEMOTRON_MODEL_FILES)) {
        const stamped = stamp.files[entry.file]
        if (!stamped || stamped.sha256 !== entry.sha256) return false

        const stats = statOf(path.join(modelDir, entry.file))
        if (!stats || stats.size !== stamped.size || stats.mtimeMs !== stamped.mtimeMs) return false
    }
    return true
}

/**
 * The authority. Returns from the stamp when it is still valid, otherwise hashes every pinned file
 * and stamps the result.
 *
 * `expectedHashes` lets the downloader pass the digests it just computed, so a fresh download is
 * stamped without reading 662 MB back off the disk a second time.
 */
export async function verifyModel(modelDir: string, expectedHashes?: { [name: string]: string }): Promise<ModelIntegrity> {
    if (!expectedHashes && isStampValid(modelDir)) return "ok"

    const files: { [name: string]: StampedFile } = {}
    let outdated = false

    for (const entry of Object.values(NEMOTRON_MODEL_FILES)) {
        const file = path.join(modelDir, entry.file)
        const stats = statOf(file)
        if (!stats?.isFile()) return "missing"

        let digest: string
        try {
            digest = expectedHashes?.[entry.file] ?? (await sha256File(file))
        } catch {
            return "missing"
        }

        // keep going rather than returning early: stamping a partially checked model would let a
        // later cheap check trust files this pass never actually read
        if (digest !== entry.sha256) outdated = true
        files[entry.file] = { sha256: digest, size: stats.size, mtimeMs: stats.mtimeMs }
    }

    if (outdated) return "outdated"

    writeStamp(modelDir, files)
    return "ok"
}
