// AI AUTO SCRIPTURE - keep the microphone audio of a session, on request.
//
// Transcription failures reported from live services have repeatedly failed to reproduce from
// recorded sermons: full 47-minute messages, worship, and silence all decode cleanly on the bench
// while a live session shows the decoder locking into a cycle. The difference is the input itself -
// a real microphone in a real room, with no noise suppression, running for hours - and no amount of
// reasoning about it substitutes for having the actual samples.
//
// So this writes exactly what the engine received, in the format it received it, straight to a WAV.
// The result drops into the benchmark as a fixture and turns "it happens live" into something that
// can be run, measured and fixed.
//
// Off unless asked for. It is a microphone recording of a church service: it stays on the machine
// that made it, and only exists while someone is deliberately chasing a fault.

import fs from "fs"
import path from "path"

const SAMPLE_RATE = 16000
/** Stop rather than fill a disk: ~3 hours of 16 kHz mono s16. */
const MAX_BYTES = 350 * 1024 * 1024

export class SessionAudioRecorder {
    private handle: number | null = null
    private bytes = 0
    private file = ""

    /** Begin a recording. Failure is logged and ignored - a diagnostic must never break a service. */
    start(directory: string, startedAtMs: number): void {
        try {
            fs.mkdirSync(directory, { recursive: true })
            this.file = path.join(directory, `session-${new Date(startedAtMs).toISOString().replace(/[:.]/g, "-")}.wav`)
            this.handle = fs.openSync(this.file, "w")
            // a placeholder header; the sizes are only known once the session ends
            fs.writeSync(this.handle, wavHeader(0))
            this.bytes = 0
            console.info(`[ai] recording session audio to ${this.file}`)
        } catch (err) {
            console.error("[ai] Could not start the session audio recording:", err)
            this.handle = null
        }
    }

    write(buffer: Uint8Array): void {
        if (this.handle === null) return
        if (this.bytes + buffer.byteLength > MAX_BYTES) {
            console.warn("[ai] session audio recording reached its size limit - stopping it")
            this.stop()
            return
        }

        try {
            fs.writeSync(this.handle, buffer)
            this.bytes += buffer.byteLength
        } catch (err) {
            console.error("[ai] Session audio recording failed:", err)
            this.stop()
        }
    }

    /** Close the file, writing the real sizes into the header now that they are known. */
    stop(): string | null {
        if (this.handle === null) return null

        const handle = this.handle
        this.handle = null
        try {
            fs.writeSync(handle, wavHeader(this.bytes), 0, 44, 0)
            fs.closeSync(handle)
            console.info(`[ai] session audio written: ${this.file} (${(this.bytes / SAMPLE_RATE / 2 / 60).toFixed(1)} minutes)`)
            return this.file
        } catch (err) {
            console.error("[ai] Could not finish the session audio recording:", err)
            return null
        }
    }

    get active(): boolean {
        return this.handle !== null
    }
}

/** 16 kHz mono s16 - the format the renderer already sends, so nothing is converted. */
function wavHeader(dataBytes: number): Buffer {
    const header = Buffer.alloc(44)
    header.write("RIFF", 0, "ascii")
    header.writeUInt32LE(36 + dataBytes, 4)
    header.write("WAVE", 8, "ascii")
    header.write("fmt ", 12, "ascii")
    header.writeUInt32LE(16, 16)
    header.writeUInt16LE(1, 20)
    header.writeUInt16LE(1, 22)
    header.writeUInt32LE(SAMPLE_RATE, 24)
    header.writeUInt32LE(SAMPLE_RATE * 2, 28)
    header.writeUInt16LE(2, 32)
    header.writeUInt16LE(16, 34)
    header.write("data", 36, "ascii")
    header.writeUInt32LE(dataBytes, 40)
    return header
}
