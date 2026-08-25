// AI BENCH - replay a live session recording (audioRecorder.ts) through the production stream
// driver at max speed, logging every final and every interim change with audio timestamps.
// This is the tool that turns "it looked wrong live" into a diffable event log:
//
//   PROBE_WAV=".../bin/bench/sessions/session-<ts>.wav" PROBE_OUT=/tmp/replay.log \
//     npx vitest run --config config/testing/vitest.config.ts src/electron/ai/speech/bench/sessionReplay.test.ts
//
// The recording is byte-exactly what the engine heard, so a replay reproduces the live decode -
// it caught the VAD-miss lump (22s of preaching over a music bed committing as one block) that
// three weeks of synthetic fixtures never showed.
import fs from "fs"
import { describe, it } from "vitest"
import { createDriver } from "./engines"
import { readWav16kMono, floatToInt16 } from "./wav"

const FILE = process.env.PROBE_WAV || ""
const describeIf = FILE && fs.existsSync(FILE) ? describe : describe.skip

describeIf("session replay probe", () => {
    it("replays and logs emissions", async () => {
        const wav = readWav16kMono(FILE)
        const out: string[] = []
        let lastInterim = ""
        let pushedMs = 0

        const driver = createDriver(
            { id: "probe", engine: "nemotron", decode: "stream" },
            {
                onSegment: (segment) => {
                    out.push(`${(pushedMs / 1000).toFixed(1).padStart(7)}  FINAL(${segment.utteranceEnd ? "end" : "mid"})  ${segment.text}`)
                },
                onInterim: (text) => {
                    if (text === lastInterim) return
                    lastInterim = text
                    out.push(`${(pushedMs / 1000).toFixed(1).padStart(7)}  interim     ${text}`)
                },
                onError: (message) => out.push(`ERROR ${message}`)
            }
        )

        await driver.start()
        const CHUNK = 1600 // 100ms
        for (let i = 0; i < wav.samples.length; i += CHUNK) {
            const slice = wav.samples.subarray(i, Math.min(i + CHUNK, wav.samples.length))
            const int16 = floatToInt16(slice)
            driver.pushAudio(new Uint8Array(int16.buffer, int16.byteOffset, int16.byteLength))
            pushedMs += (slice.length / 16000) * 1000
        }
        await driver.stop()

        fs.writeFileSync(process.env.PROBE_OUT || "/tmp/probe.log", out.join("\n"))
        console.log(`wrote ${out.length} events`)
    }, 1200000)
})
