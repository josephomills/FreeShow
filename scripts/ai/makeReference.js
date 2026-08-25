#!/usr/bin/env node
// AI BENCH - pseudo-reference transcripts for fixtures nobody has hand-typed.
//
// makeFixtures.js can produce sermon fixtures in minutes, but it cannot produce a transcript, so
// those fixtures measure latency and decode cost and nothing else - WER needs a reference. Typing
// two minutes of preaching by hand takes about twenty, which is why the sermon manifests have sat
// at timingSource "none" with no transcript at all.
//
// So this transcribes them with a much stronger OFFLINE model (whisper.cpp large-v3, batch, beam
// search, whole file at once) than anything the app could run live, and writes the result back
// into the manifest. That is NOT ground truth and this script never pretends otherwise:
//
//   * A pseudo-reference is only meaningful for comparing OTHER engines AGAINST EACH OTHER. The
//     absolute WER it produces is "distance from whisper", not "distance from what was said", and
//     whisper's own errors become every other engine's errors for free.
//   * Scoring whisper against a whisper-made reference is circular and yields a flattering number
//     that means nothing. --for whisper is refused, not warned about.
//   * Every entry it writes is stamped approximate, with the model and tool that made it, so a
//     reader of the manifest can never mistake it for a transcript a human checked.
//
// Hand-correcting the output is strictly better than regenerating it, and cheap - fixing whisper
// is far quicker than typing from nothing. Once a transcript has been edited by hand this script
// notices (the stamp carries a hash of what it wrote) and refuses to overwrite it without --force.
//
//   node scripts/ai/makeReference.js --set sermons --model large-v3 --for nemotron
//   node scripts/ai/makeReference.js --set sermons --fixture 01-01-bible-study-part-1 --dry-run

const { execFileSync } = require("child_process")
const { createHash } = require("crypto")
const fs = require("fs")
const os = require("os")
const path = require("path")

// best first: the point of an offline reference is accuracy, and the runtime cost of large-v3 on a
// two-minute clip is irrelevant when it runs once per fixture, ever
const MODEL_PREFERENCE = ["large-v3", "large-v2", "large", "medium", "medium.en", "small", "small.en", "base", "base.en", "tiny", "tiny.en"]

// the tool that makes the reference cannot also be the engine under test - see the header
const CIRCULAR_ENGINES = ["whisper", "whisper.cpp", "whispercpp", "ggml"]

// a download interrupted part-way leaves a short ggml-*.bin behind, which whisper would only
// reject after the operator has waited for it to start
const MIN_MODEL_BYTES = 1024 * 1024

function parseArgs(argv) {
    const args = { set: "sermons", manifest: null, model: null, whisper: null, language: "en", fixture: null, date: new Date().toISOString().slice(0, 10), for: null, dryRun: false, force: false }
    for (let i = 0; i < argv.length; i++) {
        const key = argv[i].replace(/^--/, "")
        const value = argv[i + 1]
        if (key === "dry-run") {
            args.dryRun = true
            continue
        }
        if (key === "force") {
            args.force = true
            continue
        }
        if (key === "set") args.set = value
        else if (key === "manifest") args.manifest = value
        else if (key === "model") args.model = value
        else if (key === "whisper") args.whisper = value
        else if (key === "language") args.language = value
        else if (key === "fixture") args.fixture = value
        else if (key === "date") args.date = value
        else if (key === "for") args.for = value
        else continue
        i++
    }
    return args
}

function expandHome(target) {
    return target.startsWith("~") ? path.join(os.homedir(), target.slice(1)) : target
}

function resolveUserDataDir() {
    if (process.env.FREESHOW_USER_DATA) return process.env.FREESHOW_USER_DATA

    const home = os.homedir()
    return process.platform === "darwin" ? path.join(home, "Library", "Application Support", "FreeShow") : process.platform === "win32" ? path.join(process.env.APPDATA || path.join(home, "AppData", "Roaming"), "FreeShow") : path.join(process.env.XDG_CONFIG_HOME || path.join(home, ".config"), "FreeShow")
}

function resolveFixtureRoot() {
    if (process.env.FREESHOW_AI_FIXTURES) return process.env.FREESHOW_AI_FIXTURES
    return path.join(resolveUserDataDir(), "bin", "bench", "fixtures")
}

/** The app's own whisper model directory - a reference should use a model the operator already trusts. */
function modelsDir() {
    return path.join(resolveUserDataDir(), "bin", "whisper", "models")
}

function presentModels() {
    try {
        return fs
            .readdirSync(modelsDir())
            .filter((file) => file.startsWith("ggml-") && file.endsWith(".bin"))
            .map((file) => ({ id: file.slice(5, -4), file: path.join(modelsDir(), file), bytes: fs.statSync(path.join(modelsDir(), file)).size }))
            .filter((model) => model.bytes > MIN_MODEL_BYTES)
    } catch {
        return []
    }
}

function pickModel(models, requested) {
    if (requested) {
        // the id becomes a file path, so refuse anything that could climb out of the models dir
        if (!/^[a-zA-Z0-9][a-zA-Z0-9._-]*$/.test(requested)) throw new Error(`invalid model id "${requested}"`)
        return models.find((model) => model.id === requested) || null
    }
    for (const id of MODEL_PREFERENCE) {
        const found = models.find((model) => model.id === id)
        if (found) return found
    }
    return models[0] || null
}

/** An explicit --whisper never falls back to another binary: silently benchmarking a different build is worse than stopping. */
function resolveWhisper(explicit) {
    const candidates = explicit ? [explicit] : [process.env.FREESHOW_WHISPER_BIN, "/opt/homebrew/bin/whisper-cli", "/usr/local/bin/whisper-cli", "whisper-cli"].filter(Boolean)

    for (const candidate of candidates.map(expandHome)) {
        try {
            execFileSync(candidate, ["--version"], { stdio: "ignore" })
            return candidate
        } catch {
            continue
        }
    }
    return null
}

function whisperVersion(binary) {
    try {
        const out = execFileSync(binary, ["--version"], { encoding: "utf8", stdio: ["ignore", "pipe", "ignore"] })
        const match = out.match(/version:?\s*([\w.\-+]+)/i)
        return match ? match[1] : out.trim().split("\n").pop().trim() || "unknown"
    } catch {
        return "unknown"
    }
}

/**
 * No initial --prompt on purpose. Biasing whisper towards book names would make the reference
 * agree with the scripture vocabulary the detection layer is being measured on, which is the same
 * circularity as scoring whisper with whisper, one layer down.
 */
function decodeArgs(language) {
    return ["-l", language, "-bs", "5", "-bo", "5", "-np", "-nt"]
}

/** Non-speech annotations are whisper's commentary, not words anyone said - as reference they would score every engine a deletion. */
function cleanTranscript(raw) {
    return raw
        .replace(/\[[^\]]*\]/g, " ")
        .replace(/\((?:[^)]*\b(?:music|laughter|applause|silence|inaudible)\b[^)]*)\)/gi, " ")
        .replace(/\s+/g, " ")
        .trim()
}

function transcribe(binary, model, language, audioPath) {
    const out = execFileSync(binary, ["-m", model.file, ...decodeArgs(language), "-f", audioPath], { encoding: "utf8", stdio: ["ignore", "pipe", "inherit"], maxBuffer: 64 * 1024 * 1024 })
    return cleanTranscript(out)
}

/** 44 bytes is a bare WAV header - makeFixtures.js can leave a stub behind if ffmpeg died. */
function audioReady(file) {
    try {
        return fs.existsSync(file) && fs.statSync(file).size > 44
    } catch {
        return false
    }
}

function textHash(text) {
    return createHash("sha256").update(text).digest("hex").slice(0, 12)
}

/**
 * Ours to replace, or someone's work to protect? Only a stamp whose hash still matches the text
 * proves nothing has been edited since this script wrote it. Everything else - a transcript with
 * no stamp, or one stamped by some other tool - is treated as a human's until proven otherwise,
 * because destroying a hand-correction is far worse than making the operator type --force.
 */
function protectedReason(entry) {
    if (!entry.transcript) return null

    const stamp = entry.reference
    if (stamp && stamp.kind === "pseudo") return stamp.textHash === textHash(entry.transcript) ? null : "hand-corrected since this script wrote it"
    if (entry.transcriptApproximate) return "machine-made by another tool, with no hash to check for later edits"
    return "hand-written"
}

function stampEntry(entry, text, model, args, tool) {
    entry.transcript = text
    entry.transcriptApproximate = true
    entry.transcriptSource = `whisper.cpp ${model.id}, ${args.date}`
    entry.reference = {
        kind: "pseudo",
        approximate: true,
        tool: `whisper.cpp ${tool}`,
        model: model.id,
        modelBytes: model.bytes,
        language: args.language,
        decodeArgs: decodeArgs(args.language),
        date: args.date,
        textHash: textHash(text),
        invalidFor: ["whisper"],
        caveat: "Machine-made pseudo-reference, NOT ground truth. Valid only for comparing other engines against each other; circular for whisper. Hand-correct before quoting an absolute WER."
    }

    // transcriptApproximate and reference are extra JSON, which loadFixtureSet carries through but
    // the Fixture type does not name - so the same warning goes in notes, which every consumer sees
    const marker = `APPROXIMATE transcript: whisper.cpp ${model.id}, ${args.date} - not hand-checked.`
    const existing = (entry.notes || "").replace(/\s*APPROXIMATE transcript:.*?hand-checked\./g, "").trim()
    entry.notes = existing ? `${existing.replace(/[.\s]*$/, "")}. ${marker}` : marker

    // a transcript makes WER measurable, not latency - expected[] and the phraseEndMs anchors in it
    // are untouched, so whatever provenance the timings already had is still exactly true of them
    if (!entry.timingSource) entry.timingSource = "none"
}

function warnBanner() {
    console.log(`\n${"!".repeat(78)}`)
    console.log(`PSEUDO-REFERENCE - NOT GROUND TRUTH`)
    console.log(`A transcript written by one ASR model is not what was said, it is what that model`)
    console.log(`heard. Its mistakes become every other engine's mistakes, so the absolute WER this`)
    console.log(`produces is meaningless on its own. Use it to RANK engines against each other, and`)
    console.log(`never to score whisper, which would be scoring a model against itself.`)
    console.log(`Hand-correcting the output is quick and makes the number real. Prefer it.`)
    console.log(`${"!".repeat(78)}\n`)
}

function main() {
    const args = parseArgs(process.argv.slice(2))

    if (args.for && CIRCULAR_ENGINES.some((engine) => args.for.toLowerCase().includes(engine))) {
        console.error(`refusing: --for "${args.for}" names the same engine that would write the reference.`)
        console.error(`Scoring whisper against a whisper-made transcript measures nothing - it is circular by construction.`)
        console.error(`Hand-correct a transcript (or type one) if whisper itself is what you need to measure.`)
        process.exit(1)
    }

    const manifestPath = args.manifest ? expandHome(args.manifest) : path.join(resolveFixtureRoot(), `${args.set}.json`)
    if (!fs.existsSync(manifestPath)) {
        console.error(`no manifest at ${manifestPath}`)
        console.error(`usage: makeReference.js [--set sermons | --manifest <path>] [--model large-v3] [--fixture <id>] [--language en]`)
        console.error(`                        [--for <engines you mean to score>] [--date YYYY-MM-DD] [--whisper <path>] [--dry-run] [--force]`)
        process.exit(1)
    }

    const models = presentModels()
    let model = null
    try {
        model = pickModel(models, args.model)
    } catch (err) {
        console.error(err.message)
        process.exit(1)
    }
    if (!model) {
        const present = models.length ? models.map((m) => `${m.id} (${(m.bytes / 1e6).toFixed(0)} MB)`).join(", ") : "none"
        console.error(args.model ? `whisper model "${args.model}" is not in ${modelsDir()}` : `no whisper model in ${modelsDir()}`)
        console.error(`present: ${present}`)
        console.error(`Download one in FreeShow (AI settings), or place ggml-<id>.bin there. large-v3 is what a reference wants.`)
        process.exit(1)
    }

    const binary = resolveWhisper(args.whisper)
    if (!binary && !args.dryRun) {
        console.error(args.whisper ? `not a working whisper binary: ${args.whisper}` : `whisper-cli not found (brew install whisper-cpp, or --whisper <path>)`)
        process.exit(1)
    }
    const tool = binary ? whisperVersion(binary) : "not-run"

    warnBanner()

    const manifest = JSON.parse(fs.readFileSync(manifestPath, "utf8"))
    const root = resolveFixtureRoot()
    const selected = manifest.fixtures.filter((entry) => !args.fixture || entry.id === args.fixture)
    if (!selected.length) {
        console.error(`no fixture "${args.fixture}" in ${manifestPath}`)
        process.exit(1)
    }

    console.log(`${manifestPath}`)
    console.log(`model: ${model.id}  tool: whisper.cpp ${tool}  language: ${args.language}  date: ${args.date}${args.dryRun ? "  (dry run)" : ""}\n`)

    let written = 0
    let failed = 0
    for (const entry of selected) {
        const audioPath = path.isAbsolute(entry.file) ? entry.file : path.join(root, entry.file)
        if (!audioReady(audioPath)) {
            console.log(`  ${entry.id}: audio missing or empty (${audioPath}) - skipped`)
            continue
        }
        const protectedBy = protectedReason(entry)
        if (protectedBy && !args.force) {
            console.log(`  ${entry.id}: transcript ${protectedBy} - kept (--force to overwrite)`)
            continue
        }
        if (args.dryRun) {
            const replacing = entry.transcript ? "would replace its transcript" : "would add a transcript"
            console.log(`  ${entry.id}: ${replacing}`)
            continue
        }

        const started = Date.now()
        let text = ""
        try {
            text = transcribe(binary, model, args.language, audioPath)
        } catch (err) {
            // one bad clip should not throw away the decodes that already succeeded
            failed++
            console.log(`  ${entry.id}: whisper exited ${err.status ?? "abnormally"} - see the error above`)
            continue
        }
        if (!text) {
            failed++
            console.log(`  ${entry.id}: whisper returned nothing - skipped`)
            continue
        }
        stampEntry(entry, text, model, args, tool)
        written++
        console.log(`  ${entry.id}: ${text.split(" ").length} words in ${((Date.now() - started) / 1000).toFixed(0)}s`)
        console.log(`    ${text.slice(0, 100)}${text.length > 100 ? "..." : ""}`)
    }

    if (args.dryRun) {
        console.log(`\ndry run - ${manifestPath} untouched`)
        return
    }
    if (failed) process.exitCode = 1
    if (!written) {
        console.log(`\nnothing written`)
        return
    }

    fs.writeFileSync(manifestPath, JSON.stringify(manifest, null, 4))
    console.log(`\n${written} transcript(s) -> ${manifestPath}`)
    console.log(`Each is marked transcriptApproximate: true. WER against it ranks engines; it does not measure them.`)
    console.log(`Read them, fix what whisper got wrong, and the stamp will stop this script overwriting your edits.`)
}

main()
