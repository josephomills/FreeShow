#!/usr/bin/env node
// AI BENCH - make a clean recording sound like it was played into a microphone.
//
// A fault reported from live use turned out to be a sermon played through speakers and picked up by
// the laptop's own microphone. That is not the same signal as the file: it arrives with the room's
// reverb, the comb filtering of a direct and reflected path arriving together, the speaker's and
// microphone's own responses, and a far worse signal-to-noise ratio.
//
// Every fixture here so far has been the clean file, which is why nothing reproduced. A greedy
// RNN-T has no beam to recover with, so what it does under acoustic uncertainty is exactly the
// behaviour worth measuring - and it cannot be measured on audio that is never uncertain.
//
//   node scripts/ai/degradeFixture.js --set longform --snr 10 --out longform-room
//
// --snr is in dB against the speech: 20 is a quiet room, 10 a normal one, 5 poor.

const { execFileSync } = require("child_process")
const fs = require("fs")
const os = require("os")
const path = require("path")

function parseArgs(argv) {
    const args = { set: "longform", snr: 10, out: null, reverb: true }
    for (let i = 0; i < argv.length; i++) {
        const key = argv[i].replace(/^--/, "")
        if (key === "set") args.set = argv[++i]
        else if (key === "snr") args.snr = Number(argv[++i])
        else if (key === "out") args.out = argv[++i]
        else if (key === "no-reverb") args.reverb = false
    }
    args.out = args.out || `${args.set}-room`
    return args
}

function fixtureRoot() {
    if (process.env.FREESHOW_AI_FIXTURES) return process.env.FREESHOW_AI_FIXTURES
    const home = os.homedir()
    const base =
        process.platform === "darwin"
            ? path.join(home, "Library", "Application Support", "FreeShow")
            : path.join(process.env.APPDATA || path.join(home, ".config"), "FreeShow")
    return path.join(base, "bin", "bench", "fixtures")
}

/**
 * Speaker and microphone both roll off the extremes, the room adds early reflections and a tail,
 * and the noise floor sits well above a mastered file's. Built from ffmpeg filters rather than a
 * measured impulse response so it needs nothing but ffmpeg - it is a caricature of a room, but it
 * is uncertain in the ways a room is, which is the property being tested.
 */
function filterChain(snrDb, reverb) {
    const speech = "highpass=f=90,lowpass=f=7500"
    const room = reverb ? ",aecho=0.85:0.7:22|41|73|113:0.32|0.24|0.16|0.09" : ""
    // noise level relative to speech; ffmpeg's anoisesrc takes an amplitude, so convert from dB
    const amplitude = Math.max(0.001, Math.pow(10, -snrDb / 20) * 0.5).toFixed(4)
    return { speech: `${speech}${room},dynaudnorm=g=7`, amplitude }
}

function main() {
    const args = parseArgs(process.argv.slice(2))
    const root = fixtureRoot()
    const manifestPath = path.join(root, `${args.set}.json`)
    if (!fs.existsSync(manifestPath)) {
        console.error(`no manifest at ${manifestPath}`)
        process.exit(1)
    }

    const manifest = JSON.parse(fs.readFileSync(manifestPath, "utf8"))
    const outDir = path.join(root, args.out)
    fs.mkdirSync(outDir, { recursive: true })

    const { speech, amplitude } = filterChain(args.snr, args.reverb)
    const fixtures = []

    for (const fixture of manifest.fixtures) {
        const source = path.isAbsolute(fixture.file) ? fixture.file : path.join(root, fixture.file)
        if (!fs.existsSync(source)) continue

        const relative = path.join(args.out, `${fixture.id}.wav`)
        const target = path.join(root, relative)
        console.log(`  ${fixture.id} -> SNR ${args.snr} dB${args.reverb ? " + room" : ""}`)

        execFileSync(
            "ffmpeg",
            ["-y", "-v", "error", "-i", source, "-filter_complex", `anoisesrc=c=pink:a=${amplitude}:r=16000[n];[0:a]${speech}[s];[s][n]amix=inputs=2:duration=first:weights=1 1[out]`, "-map", "[out]", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", target],
            { maxBuffer: 1 << 28 }
        )

        fixtures.push({ ...fixture, id: `${fixture.id}-room`, file: relative, snrDb: args.snr, tier: "degraded", notes: `${fixture.id} played into a microphone: room reflections, band limiting and pink noise at ${args.snr} dB SNR` })
    }

    fs.writeFileSync(path.join(root, `${args.out}.json`), JSON.stringify({ id: `${args.out}-local`, description: `${manifest.id} as heard through speakers and a microphone, ${args.snr} dB SNR. Clean files never reproduced the decoder cycles seen live; this is the acoustic path that was missing.`, fixtures }, null, 4))

    console.log(`\n${fixtures.length} degraded fixtures -> ${path.join(root, `${args.out}.json`)}`)
}

main()
