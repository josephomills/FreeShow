import fs from "fs"
import path from "path"
import { ToMain } from "../../../../types/IPC/ToMain"
import { sendToMain } from "../../../IPC/main"
import { DownloadManager } from "../DownloadManager"
import { MODEL_BASE_URL, NEMOTRON_MODEL_BYTES, NEMOTRON_MODEL_FILES, NEMOTRON_VAD_FILE, VAD_MODEL_SHA256, VAD_MODEL_URL } from "./nemotronFiles"

// the pinned file table and source URLs live in nemotronFiles.ts, so the runtime loader and the
// benchmark harness can read them without importing this module's electron-dependent downloader
export { NEMOTRON_MODEL_FILES, NEMOTRON_MODEL_BYTES, NEMOTRON_VAD_FILE } from "./nemotronFiles"

export class NemotronSetupManager {
    static getBinaryName() {
        return process.platform === "win32" ? "nemotron-cli.exe" : "nemotron-cli"
    }

    static engineDLM: DownloadManager | null = null
    static getDownloadManager() {
        if (!this.engineDLM) this.engineDLM = new DownloadManager("nemotron", "Nemotron model")
        return this.engineDLM
    }

    static async downloadEngine(outputFolder: string) {
        const dlm = this.getDownloadManager()

        const jobs = [...Object.values(NEMOTRON_MODEL_FILES).map((entry) => ({ url: `${MODEL_BASE_URL}/${entry.file}`, file: entry.file, sha256: entry.sha256 })), { url: VAD_MODEL_URL, file: NEMOTRON_VAD_FILE, sha256: VAD_MODEL_SHA256 }]

        // one download spans several files, so progress is reported against the known total rather than per file
        let completedBytes = 0
        for (const job of jobs) {
            const target = path.join(outputFolder, job.file)

            // a file from an earlier run only counts when its checksum proves it is exactly the pinned content
            if (await this.verifyEngine(target)) {
                if ((await dlm.computeSha256(target)) === job.sha256) {
                    completedBytes += fs.statSync(target).size
                    continue
                }
                fs.unlinkSync(target)
            }

            const base = completedBytes
            try {
                await dlm.downloadFile(job.url, target, {
                    // one stable key for the whole multi-file download, so the renderer shows a single progress entry
                    onProgress: (bytes) => {
                        sendToMain(ToMain.MEDIA_DOWNLOAD_PROGRESS, { url: dlm.key, name: dlm.name, progress: base + bytes, total: NEMOTRON_MODEL_BYTES, status: "downloading" })
                    }
                })

                // integrity check against the pinned hash - a corrupt or substituted file must never land
                if ((await dlm.computeSha256(target)) !== job.sha256) {
                    fs.unlinkSync(target)
                    throw new Error(`Downloaded ${job.file} failed checksum verification`)
                }
            } catch (err) {
                if (dlm.isAbortError(err)) return { ok: false, error: "Download was cancelled." }
                return dlm.reportError(`Failed to download Nemotron model: ${dlm.errorMessage(err)}`)
            }
            completedBytes = base + fs.statSync(target).size
        }

        return dlm.reportComplete()
    }

    static cancelEngineDownload() {
        const dlm = this.getDownloadManager()
        if (dlm.isDownloading()) dlm.cancel()
    }

    static async verifyEngine(binaryPath: string) {
        if (!binaryPath) return false

        try {
            return fs.existsSync(binaryPath) && fs.statSync(binaryPath).size > 1024
        } catch {
            return false
        }
    }

    // this has just one model, treated as "engine"
    static async downloadModel(_modelId: string, _outputPath: string) {
        return false
    }
    static cancelModelDownload(_modelId: string) {
        return true
    }
    static async verifyModel(_filePath: string): Promise<boolean> {
        return false
    }
}
