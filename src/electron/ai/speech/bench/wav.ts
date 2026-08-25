// AI BENCH - minimal WAV reader for the speech benchmark harness.
// The inverse of buildWavBuffer() in ../whisper/transcriber.ts: that one writes the canonical
// 44-byte header, this one has to cope with whatever a real fixture file carries (LIST/INFO
// chunks from an editor, a WAVE_FORMAT_EXTENSIBLE fmt block, cue points). So it walks the chunk
// table instead of assuming offset 44 - the same lesson FluidAudio's docs spell out: hand-parsing
// a WAV by fixed offsets yields garbage samples, which show up downstream as empty transcripts.

import fs from "fs"

export const BENCH_SAMPLE_RATE = 16000

export interface WavData {
    /** Interleaved samples, already downmixed to mono and resampled to BENCH_SAMPLE_RATE. */
    samples: Int16Array
    /** Sample rate of `samples` - always BENCH_SAMPLE_RATE after readWav16kMono(). */
    sampleRate: number
    durationMs: number
}

const FORMAT_PCM = 1
const FORMAT_FLOAT = 3
const FORMAT_EXTENSIBLE = 0xfffe

interface RawWav {
    channels: number
    sampleRate: number
    bitsPerSample: number
    format: number
    data: Buffer
}

/** Walk the RIFF chunk table and return the fmt/data pair. Throws with a specific reason. */
function parseRiff(buffer: Buffer): RawWav {
    if (buffer.length < 12) throw new Error("WAV too short to hold a RIFF header")
    if (buffer.toString("ascii", 0, 4) !== "RIFF") throw new Error("not a RIFF file")
    if (buffer.toString("ascii", 8, 12) !== "WAVE") throw new Error("RIFF file is not WAVE")

    let channels = 0
    let sampleRate = 0
    let bitsPerSample = 0
    let format = 0
    let data: Buffer | null = null

    let offset = 12
    while (offset + 8 <= buffer.length) {
        const id = buffer.toString("ascii", offset, offset + 4)
        const size = buffer.readUInt32LE(offset + 4)
        const body = offset + 8
        // a truncated final chunk is common in recordings cut mid-write - take what is there
        const end = Math.min(body + size, buffer.length)

        if (id === "fmt ") {
            if (end - body < 16) throw new Error("fmt chunk is truncated")
            format = buffer.readUInt16LE(body)
            channels = buffer.readUInt16LE(body + 2)
            sampleRate = buffer.readUInt32LE(body + 4)
            bitsPerSample = buffer.readUInt16LE(body + 14)
            // WAVE_FORMAT_EXTENSIBLE hides the real format in the first 2 bytes of the GUID
            if (format === FORMAT_EXTENSIBLE && end - body >= 26) format = buffer.readUInt16LE(body + 24)
        } else if (id === "data") {
            data = buffer.subarray(body, end)
        }

        // chunks are word-aligned: an odd size is followed by a pad byte
        offset = body + size + (size % 2)
    }

    if (!data) throw new Error("no data chunk")
    if (!channels || !sampleRate || !bitsPerSample) throw new Error("no usable fmt chunk")
    return { channels, sampleRate, bitsPerSample, format, data }
}

/** Decode the data chunk to float samples in [-1, 1], keeping the channel interleaving. */
function decodeSamples(raw: RawWav): Float32Array {
    const { data, bitsPerSample, format } = raw

    if (format === FORMAT_FLOAT && bitsPerSample === 32) {
        const count = Math.floor(data.length / 4)
        const out = new Float32Array(count)
        for (let i = 0; i < count; i++) out[i] = data.readFloatLE(i * 4)
        return out
    }

    if (format !== FORMAT_PCM) throw new Error(`unsupported WAV format ${format} (only PCM and 32-bit float)`)

    if (bitsPerSample === 16) {
        const count = Math.floor(data.length / 2)
        const out = new Float32Array(count)
        for (let i = 0; i < count; i++) out[i] = data.readInt16LE(i * 2) / 32768
        return out
    }
    if (bitsPerSample === 24) {
        const count = Math.floor(data.length / 3)
        const out = new Float32Array(count)
        for (let i = 0; i < count; i++) {
            const o = i * 3
            // sign-extend the 24-bit little-endian value
            const value = (data[o] | (data[o + 1] << 8) | (data[o + 2] << 16)) << 8
            out[i] = value / 8 / 8388608
        }
        return out
    }
    if (bitsPerSample === 32) {
        const count = Math.floor(data.length / 4)
        const out = new Float32Array(count)
        for (let i = 0; i < count; i++) out[i] = data.readInt32LE(i * 4) / 2147483648
        return out
    }
    if (bitsPerSample === 8) {
        // 8-bit PCM in WAV is unsigned
        const out = new Float32Array(data.length)
        for (let i = 0; i < data.length; i++) out[i] = (data[i] - 128) / 128
        return out
    }
    throw new Error(`unsupported bit depth ${bitsPerSample}`)
}

/** Average the channels down to one. */
export function downmixToMono(interleaved: Float32Array, channels: number): Float32Array {
    if (channels <= 1) return interleaved

    const frames = Math.floor(interleaved.length / channels)
    const out = new Float32Array(frames)
    for (let frame = 0; frame < frames; frame++) {
        let sum = 0
        for (let channel = 0; channel < channels; channel++) sum += interleaved[frame * channels + channel]
        out[frame] = sum / channels
    }
    return out
}

/**
 * Linear resample. Deliberately NOT the renderer's path: capture uses Chromium's sinc resampler
 * (frontend/ai/stt/stt.ts sets AudioContext sampleRate and lets it do the work), which is better
 * than this. Fixtures should therefore be authored at 16 kHz so this never runs - it exists so a
 * stray 44.1 kHz file degrades to a warning rather than a crash. A resampled fixture is not a
 * valid basis for a WER comparison against one that was not; the runner flags it.
 */
export function resampleLinear(samples: Float32Array, from: number, to: number): Float32Array {
    if (from === to) return samples

    const ratio = from / to
    const count = Math.max(1, Math.floor(samples.length / ratio))
    const out = new Float32Array(count)
    for (let i = 0; i < count; i++) {
        const position = i * ratio
        const index = Math.floor(position)
        const frac = position - index
        const a = samples[index] ?? 0
        const b = samples[index + 1] ?? a
        out[i] = a + (b - a) * frac
    }
    return out
}

export function floatToInt16(samples: Float32Array): Int16Array {
    const out = new Int16Array(samples.length)
    for (let i = 0; i < samples.length; i++) {
        const clamped = Math.max(-1, Math.min(1, samples[i]))
        out[i] = Math.round(clamped * 32767)
    }
    return out
}

/** Read a WAV file as 16 kHz mono Int16 - the exact shape the renderer sends over IPC. */
export function readWav16kMono(filePath: string): WavData & { resampledFrom?: number } {
    const raw = parseRiff(fs.readFileSync(filePath))

    const mono = downmixToMono(decodeSamples(raw), raw.channels)
    const resampled = resampleLinear(mono, raw.sampleRate, BENCH_SAMPLE_RATE)
    const samples = floatToInt16(resampled)

    return {
        samples,
        sampleRate: BENCH_SAMPLE_RATE,
        durationMs: Math.round((samples.length / BENCH_SAMPLE_RATE) * 1000),
        ...(raw.sampleRate === BENCH_SAMPLE_RATE ? {} : { resampledFrom: raw.sampleRate })
    }
}
