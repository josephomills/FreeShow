#!/usr/bin/env node
// AI BENCH - turn local audio into benchmark fixtures.
//
// Sermons are the primary use case for AI auto-scripture, so the fixtures that decide anything
// are real preaching: a room, a PA system, a preacher's cadence, and scripture references spoken
// the way people actually speak them ("Ephesians chapter two verse eight", not "Ephesians 2:8").
// Nothing about that is reproducible from read-aloud corpora.
//
// Those recordings are private and usually copyrighted, so this script keeps them at arm's
// length: audio is converted into the fixture root (outside the repo, see fixtures.ts), and the
// manifest it writes goes THERE too rather than into bench/manifests/ - the file names alone
// would identify the source. Nothing here uploads anything; ffmpeg runs locally.
//
//   node scripts/ai/makeFixtures.js --from "~/Music/The Poimano (Topical)/Messages on Acts" \
//        --count 5 --start 300 --duration 120 --set sermons
//
// Excerpts start at --start seconds in on purpose: the opening minutes of a message are usually
// announcements, worship or a prayer, and the scripture-dense teaching is further in.

const { execFileSync } = require("child_process")
const fs = require("fs")
const os = require("os")
const path = require("path")

const AUDIO_EXTENSIONS = [".mp3", ".m4a", ".wav", ".aac", ".flac", ".ogg", ".opus", ".wma"]

function parseArgs(argv) {
    const args = { count: 5, start: 300, duration: 120, set: "sermons", ext: null }
    for (let i = 0; i < argv.length; i++) {
        const key = argv[i].replace(/^--/, "")
        const value = argv[i + 1]
        if (key === "from") args.from = value
        else if (key === "count") args.count = Number(value)
        else if (key === "start") args.start = Number(value)
        else if (key === "duration") args.duration = Number(value)
        else if (key === "set") args.set = value
        else continue
        i++
    }
    return args
}

function expandHome(target) {
    return target.startsWith("~") ? path.join(os.homedir(), target.slice(1)) : target
}

function resolveFixtureRoot() {
    if (process.env.FREESHOW_AI_FIXTURES) return process.env.FREESHOW_AI_FIXTURES

    const home = os.homedir()
    const base = process.platform === "darwin" ? path.join(home, "Library", "Application Support", "FreeShow") : process.platform === "win32" ? path.join(process.env.APPDATA || path.join(home, "AppData", "Roaming"), "FreeShow") : path.join(process.env.XDG_CONFIG_HOME || path.join(home, ".config"), "FreeShow")

    return path.join(base, "bin", "bench", "fixtures")
}

function findAudio(root, out = []) {
    for (const entry of fs.readdirSync(root, { withFileTypes: true })) {
        const full = path.join(root, entry.name)
        if (entry.isDirectory()) findAudio(full, out)
        else if (AUDIO_EXTENSIONS.includes(path.extname(entry.name).toLowerCase())) out.push(full)
    }
    return out
}

function durationSeconds(file) {
    try {
        const out = execFileSync("ffprobe", ["-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", file], { encoding: "utf8" })
        return Number(out.trim()) || 0
    } catch {
        return 0
    }
}

/** Stable id from the path, so re-running produces the same fixture ids without leaking names. */
function fixtureId(file, index) {
    const stem = path
        .basename(file, path.extname(file))
        .toLowerCase()
        .replace(/[^a-z0-9]+/g, "-")
        .replace(/^-|-$/g, "")
        .slice(0, 40)
    return `${String(index + 1).padStart(2, "0")}-${stem || "clip"}`
}

function main() {
    const args = parseArgs(process.argv.slice(2))
    if (!args.from) {
        console.error("usage: makeFixtures.js --from <dir-or-file> [--count 5] [--start 300] [--duration 120] [--set sermons]")
        process.exit(1)
    }

    const from = expandHome(args.from)
    if (!fs.existsSync(from)) {
        console.error(`not found: ${from}`)
        process.exit(1)
    }

    try {
        execFileSync("ffmpeg", ["-version"], { stdio: "ignore" })
    } catch {
        console.error("ffmpeg is required (brew install ffmpeg)")
        process.exit(1)
    }

    const sources = fs.statSync(from).isDirectory() ? findAudio(from) : [from]
    if (!sources.length) {
        console.error(`no audio under ${from}`)
        process.exit(1)
    }

    // longest first: a long message is a real teaching session, a short one is often a clip or
    // an announcement, and the point is to exercise sustained continuous speech
    const chosen = sources
        .map((file) => ({ file, seconds: durationSeconds(file) }))
        .filter((entry) => entry.seconds >= args.start + args.duration)
        .sort((a, b) => b.seconds - a.seconds)
        .slice(0, args.count)

    if (!chosen.length) {
        console.error(`no file under ${from} is longer than ${args.start + args.duration}s - lower --start or --duration`)
        process.exit(1)
    }

    const root = resolveFixtureRoot()
    const outputDir = path.join(root, args.set)
    fs.mkdirSync(outputDir, { recursive: true })

    const fixtures = []
    chosen.forEach((entry, index) => {
        const id = fixtureId(entry.file, index)
        const relative = path.join(args.set, `${id}.wav`)
        const target = path.join(root, relative)

        // -ss before -i seeks by keyframe (fast); 16 kHz mono s16 matches the renderer's capture
        execFileSync("ffmpeg", ["-y", "-v", "error", "-ss", String(args.start), "-t", String(args.duration), "-i", entry.file, "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", target])

        fixtures.push({
            id,
            tier: "sermon",
            file: relative,
            durationMs: args.duration * 1000,
            timingSource: "none",
            expected: [],
            notes: `excerpt at ${args.start}s of a local recording; source not recorded here on purpose`
        })
        console.log(`  ${relative}  (${(fs.statSync(target).size / 1024 / 1024).toFixed(1)} MB)`)
    })

    const manifest = {
        id: `${args.set}-local`,
        description: `Local ${args.tier || "sermon"} excerpts, ${args.duration}s each from ${args.start}s in. Audio and this manifest live outside the repo - neither may be committed.`,
        fixtures
    }
    const manifestPath = path.join(root, `${args.set}.json`)
    fs.writeFileSync(manifestPath, JSON.stringify(manifest, null, 4))

    console.log(`\n${fixtures.length} fixtures -> ${outputDir}`)
    console.log(`manifest -> ${manifestPath}`)
    console.log(`\nNo reference transcripts yet: latency, decode cost and detection output are measurable now;`)
    console.log(`WER needs a transcript, and expected[] needs hand-marking, per fixture.`)
}

main()
