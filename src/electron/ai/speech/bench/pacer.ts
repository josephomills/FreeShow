// AI BENCH - feeds a fixture through a TranscriptionDriver the way the renderer does.
//
// The renderer's AudioWorklet (frontend/ai/stt/audioProcessor.ts) emits exactly 1600 samples
// (100 ms) of Int16 LE per message, 10 messages a second, and the driver's tuning constants are
// written against that cadence. The pacer reproduces it byte for byte, so a bench run exercises
// the same code path a live service does.
//
// It also owns THE AUDIO CLOCK, and that is the point of the file. Latency measured against the
// wall clock is a property of the machine the bench ran on; latency measured against how much
// audio has been pushed is a property of the engine. Every metric downstream reads
// `audioPushedMs()` at the moment a segment arrives, so the numbers are comparable between a
// laptop and a CI runner. Wall time is recorded too - see the note on modes below.

import type { TranscriptionDriver } from "../types"
import { percentile } from "./stats"
import { BENCH_SAMPLE_RATE } from "./wav"

/** 1600 samples @ 16 kHz - the renderer's OUTPUT_SAMPLE_COUNT. */
export const CHUNK_SAMPLES = 1600
export const CHUNK_MS = (CHUNK_SAMPLES / BENCH_SAMPLE_RATE) * 1000

export type PaceMode = "rt" | "max"

export interface PacerOptions {
    /**
     * "rt"  - one 100 ms chunk every 100 ms of wall clock. Realistic: a machine that cannot keep
     *         up shows it, because pushAudio blocks and the wall clock runs away from the audio
     *         clock. This is the only mode whose wall-time numbers mean anything.
     * "max" - push as fast as pushAudio returns. Deterministic and ~100x quicker. Audio-time
     *         latency and WER are identical to "rt"; wall-time numbers are meaningless and the
     *         runner refuses to report them.
     */
    mode: PaceMode
    /** Called after every chunk, for progress reporting on long fixtures. */
    onProgress?: (audioPushedMs: number, totalMs: number) => void
}

export interface PacerStats {
    /** Total audio handed to the driver. */
    audioPushedMs: number
    /** Wall time from first push to last. Only meaningful in "rt" mode. */
    wallMs: number
    /** Time spent inside pushAudio() itself - the driver blocking the caller. */
    pushBlockedMs: number
    pushBlockedP50: number
    pushBlockedP99: number
    pushBlockedMax: number
    /**
     * "rt" only: how far the wall clock fell behind the audio clock. A machine keeping up sits
     * near zero. Anything approaching the fixture duration means the engine cannot run live on
     * this hardware - the failure mode that audio-time latency alone would hide completely.
     */
    driftMs: number
}

/** The audio clock, shared with whoever needs to timestamp an emission. */
export class AudioClock {
    private pushedSamples = 0

    get audioPushedMs(): number {
        return (this.pushedSamples / BENCH_SAMPLE_RATE) * 1000
    }

    advance(samples: number) {
        this.pushedSamples += samples
    }
}

/** Int16Array -> the Uint8Array of Int16 LE bytes the IPC layer delivers. */
export function chunkToBytes(samples: Int16Array, start: number, count: number): Uint8Array {
    const bytes = new Uint8Array(count * 2)
    const view = new DataView(bytes.buffer)
    for (let i = 0; i < count; i++) view.setInt16(i * 2, samples[start + i], true)
    return bytes
}

const sleep = (ms: number) => new Promise<void>((resolve) => setTimeout(resolve, ms))

/**
 * Push `samples` through `driver` in renderer-shaped chunks.
 * `clock` is advanced BEFORE each pushAudio call returns to the caller's listeners, so a segment
 * emitted synchronously from inside pushAudio is timestamped with the audio that produced it.
 */
export async function pace(driver: TranscriptionDriver, samples: Int16Array, clock: AudioClock, options: PacerOptions): Promise<PacerStats> {
    const totalMs = (samples.length / BENCH_SAMPLE_RATE) * 1000
    const pushDurations: number[] = []

    const startedAt = Date.now()
    let pushBlockedMs = 0

    for (let offset = 0; offset < samples.length; offset += CHUNK_SAMPLES) {
        const count = Math.min(CHUNK_SAMPLES, samples.length - offset)
        const bytes = chunkToBytes(samples, offset, count)

        // the clock advances first: the driver may emit synchronously from inside pushAudio, and
        // that emission was caused by audio including this chunk
        clock.advance(count)

        const pushStartedAt = Date.now()
        driver.pushAudio(bytes)
        const blocked = Date.now() - pushStartedAt

        pushDurations.push(blocked)
        pushBlockedMs += blocked

        options.onProgress?.(clock.audioPushedMs, totalMs)

        if (options.mode === "rt") {
            // sleep the remainder of this chunk's wall-clock budget. A decode that overran it
            // gets no sleep at all, so the drift below accumulates honestly.
            const target = startedAt + clock.audioPushedMs
            const remaining = target - Date.now()
            if (remaining > 0) await sleep(remaining)
        } else {
            // yield so a driver that posts work to the event loop can drain between chunks
            await Promise.resolve()
        }
    }

    const wallMs = Date.now() - startedAt
    pushDurations.sort((a, b) => a - b)

    return {
        audioPushedMs: clock.audioPushedMs,
        wallMs,
        pushBlockedMs,
        pushBlockedP50: percentile(pushDurations, 50),
        pushBlockedP99: percentile(pushDurations, 99),
        pushBlockedMax: pushDurations[pushDurations.length - 1] ?? 0,
        driftMs: options.mode === "rt" ? Math.max(0, wallMs - totalMs) : 0
    }
}
