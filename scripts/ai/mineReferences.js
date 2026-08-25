#!/usr/bin/env node
// AI BENCH - find sermon excerpts that actually contain a spoken scripture reference.
//
// Detection metrics need fixtures where a reference is spoken, and those are rarer than they sound:
// of the first 19 excerpts cut at a fixed offset, only 3 contained one. Preachers announce their
// text early ("turn with me to...") and then expound it for twenty minutes without naming it again,
// so this cuts near the START of a message rather than the middle.
//
// Screening runs a small whisper model, because the question at this stage is only "does a
// reference appear here" - a fast approximate transcript answers that, and the accurate
// transcription and hand-marking are worth doing only for the clips that survive.
//
//   node scripts/ai/mineReferences.js --from "~/Music/Some Series" --count 60 --start 90 --duration 120

const { execFileSync } = require("child_process")
const fs = require("fs")
const os = require("os")
const path = require("path")

const DEFAULT_LIBRARY = path.join(os.homedir(), "Music", "The Poimano (Topical)")
const AUDIO = [".mp3", ".m4a", ".wav"]

// screening only - a book name followed by a number within a few words
const BOOKS = "genesis|exodus|leviticus|numbers|deuteronomy|joshua|judges|ruth|samuel|kings|chronicles|ezra|nehemiah|esther|job|psalm|psalms|proverbs|ecclesiastes|isaiah|jeremiah|lamentations|ezekiel|daniel|hosea|joel|amos|obadiah|jonah|micah|nahum|habakkuk|zephaniah|haggai|zechariah|malachi|matthew|mark|luke|john|acts|romans|corinthians|galatians|ephesians|philippians|colossians|thessalonians|timothy|titus|philemon|hebrews|james|peter|jude|revelation"
const NUM = "\\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred"
const REFERENCE = new RegExp(`\\b(${BOOKS})\\b[\\s,]*(chapter\\s+)?(${NUM})\\b`, "gi")

function parseArgs(argv) {
    const args = { count: 60, start: 90, duration: 120, model: "small.en", from: DEFAULT_LIBRARY }
    for (let i = 0; i < argv.length; i++) {
        const key = argv[i].replace(/^--/, "")
        if (key in args) args[key] = isNaN(Number(argv[i + 1])) ? argv[++i] : Number(argv[++i])
    }
    return args
}

function findAudio(root, out = []) {
    for (const entry of fs.readdirSync(root, { withFileTypes: true })) {
        const full = path.join(root, entry.name)
        if (entry.isDirectory()) findAudio(full, out)
        else if (AUDIO.includes(path.extname(entry.name).toLowerCase())) out.push(full)
    }
    return out
}

function userData() {
    const home = os.homedir()
    return process.platform === "darwin" ? path.join(home, "Library", "Application Support", "FreeShow") : path.join(process.env.APPDATA || path.join(home, ".config"), "FreeShow")
}

function main() {
    const args = parseArgs(process.argv.slice(2))
    const model = path.join(userData(), "bin", "whisper", "models", `ggml-${args.model}.bin`)
    if (!fs.existsSync(model)) {
        console.error(`whisper model not found: ${model}`)
        process.exit(1)
    }

    const work = fs.mkdtempSync(path.join(os.tmpdir(), "freeshow-mine-"))
    const library = args.from.startsWith("~") ? path.join(os.homedir(), args.from.slice(1)) : args.from
    if (!fs.existsSync(library)) {
        console.error(`not found: ${library}`)
        process.exit(1)
    }
    const all = findAudio(library)

    // spread across the library rather than taking a run of consecutive files, which would sample
    // one preacher, one series and one recording setup
    const step = Math.max(1, Math.floor(all.length / args.count))
    const picked = []
    for (let i = 0; i < all.length && picked.length < args.count; i += step) picked.push(all[i])

    console.log(`screening ${picked.length} of ${all.length} messages, ${args.duration}s from ${args.start}s in\n`)

    const hits = []
    picked.forEach((file, index) => {
        const wav = path.join(work, `${index}.wav`)
        try {
            execFileSync("ffmpeg", ["-y", "-v", "error", "-ss", String(args.start), "-t", String(args.duration), "-i", file, "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", wav], { stdio: "ignore" })
            execFileSync("/opt/homebrew/bin/whisper-cli", ["-m", model, "-f", wav, "-l", "en", "-otxt", "-of", wav.replace(/\.wav$/, ""), "-np", "-nf"], { stdio: "ignore" })
        } catch {
            return
        }

        const txt = wav.replace(/\.wav$/, ".txt")
        if (!fs.existsSync(txt)) return
        const text = fs.readFileSync(txt, "utf8").replace(/\s+/g, " ")

        REFERENCE.lastIndex = 0
        const found = [...text.matchAll(REFERENCE)].map((m) => m[0].trim())
        if (found.length) {
            hits.push({ file, found })
            console.log(`  HIT  ${path.basename(file)}`)
            console.log(`       ${[...new Set(found)].slice(0, 6).join(" | ")}`)
        }
        if ((index + 1) % 10 === 0) console.log(`  ... ${index + 1}/${picked.length} screened, ${hits.length} hits`)
    })

    fs.writeFileSync(path.join(work, "hits.json"), JSON.stringify({ start: args.start, duration: args.duration, hits }, null, 4))
    console.log(`\n${hits.length} of ${picked.length} clips contain a spoken reference`)
    console.log(`hits -> ${path.join(work, "hits.json")}`)
}

main()
