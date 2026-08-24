import fs from "fs"
import os from "os"
import path from "path"
import { describe, expect, it } from "vitest"
import { SessionAudioRecorder } from "./audioRecorder"

const pcm = (samples: number) => new Uint8Array(new Int16Array(samples).fill(1234).buffer)

function readHeader(file: string) {
    const buffer = fs.readFileSync(file)
    return {
        riff: buffer.toString("ascii", 0, 4),
        wave: buffer.toString("ascii", 8, 12),
        channels: buffer.readUInt16LE(22),
        sampleRate: buffer.readUInt32LE(24),
        bits: buffer.readUInt16LE(34),
        declared: buffer.readUInt32LE(40),
        actual: buffer.length - 44
    }
}

describe("session audio recorder", () => {
    const dir = () => fs.mkdtempSync(path.join(os.tmpdir(), "freeshow-rec-"))

    it("writes a WAV whose declared size matches what was written", () => {
        // the header is written before the size is known and patched at the end - if that patch is
        // wrong the file plays as silence or truncates, and the recording is worthless
        const recorder = new SessionAudioRecorder()
        const at = dir()
        recorder.start(at, 0)
        recorder.write(pcm(1600))
        recorder.write(pcm(1600))
        const file = recorder.stop()!

        const header = readHeader(file)
        expect(header.riff).toBe("RIFF")
        expect(header.wave).toBe("WAVE")
        expect(header.declared).toBe(header.actual)
        expect(header.actual).toBe(6400)
    })

    it("records the format the engine is actually given", () => {
        const recorder = new SessionAudioRecorder()
        recorder.start(dir(), 0)
        recorder.write(pcm(160))
        const header = readHeader(recorder.stop()!)

        expect(header.sampleRate).toBe(16000)
        expect(header.channels).toBe(1)
        expect(header.bits).toBe(16)
    })

    it("does nothing at all until asked", () => {
        // a diagnostic that writes without being turned on is a privacy problem, not a feature
        const recorder = new SessionAudioRecorder()
        expect(recorder.active).toBe(false)
        expect(() => recorder.write(pcm(160))).not.toThrow()
        expect(recorder.stop()).toBeNull()
    })

    it("survives a directory it cannot write to", () => {
        // a fault-finding aid must never be the thing that ends a service
        const recorder = new SessionAudioRecorder()
        recorder.start("/proc/definitely-not-writable", 0)

        expect(recorder.active).toBe(false)
        expect(() => recorder.write(pcm(160))).not.toThrow()
    })

    it("stops rather than filling the disk", () => {
        const recorder = new SessionAudioRecorder()
        recorder.start(dir(), 0)
        // one write past the cap is refused outright rather than truncated
        recorder.write(new Uint8Array(400 * 1024 * 1024))

        expect(recorder.active).toBe(false)
    })
})
