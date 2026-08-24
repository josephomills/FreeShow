#!/usr/bin/env node
// AI BENCH - download alternative sherpa-onnx model sets for comparison runs.
//
// Deliberately NOT the app's DownloadManager: these never ship, they live outside the engine's
// model directory, and pinning them to per-file SHA-256s would mean hand-maintaining a hash table
// for every tier we want to try. What IS pinned is the revision - a moving "main" would make two
// benchmark runs silently incomparable, which is worse than no benchmark at all.
//
//   node scripts/ai/getBenchModels.js en-160 multi-320
//   node scripts/ai/getBenchModels.js --all

const { spawn } = require("child_process")
const fs = require("fs")
const https = require("https")
const os = require("os")
const path = require("path")

const SETS = {
    "en-1120": "csukuangfj/sherpa-onnx-nemotron-speech-streaming-en-0.6b-int8-2026-01-14",
    "en-160": "csukuangfj2/sherpa-onnx-nemotron-speech-streaming-en-0.6b-160ms-int8-2026-04-25",
    "en-560": "csukuangfj2/sherpa-onnx-nemotron-speech-streaming-en-0.6b-560ms-int8-2026-04-25",
    "en-80": "csukuangfj2/sherpa-onnx-nemotron-speech-streaming-en-0.6b-80ms-int8-2026-04-25",
    "multi-1120": "csukuangfj2/sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-1120ms-int8-2026-06-11",
    "multi-560": "csukuangfj2/sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-560ms-int8-2026-06-11",
    "multi-320": "csukuangfj2/sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-320ms-int8-2026-06-11",
    "multi-160": "csukuangfj2/sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-160ms-int8-2026-06-11",
    // ships bpe.model, so it is the only cross-platform route to hotword biasing:
    // modified_beam_search works here and does not exist for the NeMo transducer
    "zipformer-en": "csukuangfj/sherpa-onnx-streaming-zipformer-en-2023-06-26",
    // the only other English-capable export that ships bpe.model, and newer
    "zipformer-multi": "csukuangfj/sherpa-onnx-streaming-zipformer-ar_en_id_ja_ru_th_vi_zh-2025-02-10"
}

const FILES = ["encoder.int8.onnx", "decoder.int8.onnx", "joiner.int8.onnx", "tokens.txt"]

// Nemotron exports use the plain names above. The Zipformer repos encode the training epoch and
// the streaming context in the file name, and carry several context variants, so they are matched
// by pattern and renamed on the way in - every consumer then sees the same four names.
const PATTERNS = {
    "zipformer-en": {
        "encoder.int8.onnx": /^encoder-.*chunk-16-left-128\.int8\.onnx$/,
        "decoder.int8.onnx": /^decoder-.*chunk-16-left-128\.int8\.onnx$/,
        "joiner.int8.onnx": /^joiner-.*chunk-16-left-128\.int8\.onnx$/
    },
    "zipformer-multi": {
        "encoder.int8.onnx": /^encoder-.*chunk-16-left-128\.int8\.onnx$/,
        "decoder.int8.onnx": /^decoder-.*chunk-16-left-128\.(int8\.)?onnx$/,
        "joiner.int8.onnx": /^joiner-.*chunk-16-left-128\.int8\.onnx$/
    }
}

function benchModelRoot() {
    const home = os.homedir()
    const base = process.platform === "darwin" ? path.join(home, "Library", "Application Support", "FreeShow") : process.platform === "win32" ? path.join(process.env.APPDATA || path.join(home, "AppData", "Roaming"), "FreeShow") : path.join(process.env.XDG_CONFIG_HOME || path.join(home, ".config"), "FreeShow")
    return path.join(base, "bin", "bench", "models")
}

function api(url) {
    return new Promise((resolve, reject) => {
        https
            .get(url, { headers: { "user-agent": "freeshow-bench" } }, (res) => {
                if (res.statusCode >= 300 && res.statusCode < 400 && res.headers.location) return resolve(api(res.headers.location))
                let body = ""
                res.on("data", (chunk) => (body += chunk))
                res.on("end", () => resolve(JSON.parse(body)))
            })
            .on("error", reject)
    })
}

/**
 * Shells out to curl rather than using https.get. These are 300-650 MB pulls from a CDN that
 * drops long-lived connections - the plain-node version of this function died on ETIMEDOUT and
 * "socket hang up" every time, which is the same failure the app's own DownloadManager exists to
 * handle. curl -C - resumes a .part file instead of restarting, and --retry survives the drop.
 */
function download(url, target) {
    return new Promise((resolve, reject) => {
        const part = `${target}.part`
        const child = spawn("curl", ["-L", "--fail", "--retry", "10", "--retry-delay", "2", "--retry-all-errors", "--connect-timeout", "30", "-C", "-", "--progress-bar", "-o", part, url], { stdio: ["ignore", "inherit", "inherit"] })

        child.on("error", reject)
        child.on("exit", (code) => {
            if (code !== 0) return reject(new Error(`curl exited ${code}`))
            fs.renameSync(part, target)
            resolve()
        })
    })
}

async function fetchSet(id) {
    const repo = SETS[id]
    if (!repo) throw new Error(`unknown set "${id}" - known: ${Object.keys(SETS).join(", ")}`)

    // pin the revision: "main" moving between two runs makes them silently incomparable
    const info = await api(`https://huggingface.co/api/models/${repo}`)
    const revision = info.sha
    const available = new Set(info.siblings.map((s) => s.rfilename))
    // { localName -> remoteName }; a pattern set resolves the remote name at download time
    const patterns = PATTERNS[id] || {}
    const wanted = {}
    for (const name of [...FILES, "bpe.model"]) {
        if (available.has(name)) wanted[name] = name
        else if (patterns[name]) {
            const match = [...available].find((f) => patterns[name].test(f))
            if (match) wanted[name] = match
        }
    }

    if (!Object.keys(wanted).length) throw new Error(`${repo} has none of the expected files (has: ${[...available].slice(0, 8).join(", ")})`)

    const dir = path.join(benchModelRoot(), id)
    fs.mkdirSync(dir, { recursive: true })
    fs.writeFileSync(path.join(dir, "SOURCE.json"), JSON.stringify({ id, repo, revision, files: wanted }, null, 4))

    console.log(`\n${id}  <-  ${repo}@${revision.slice(0, 8)}`)
    for (const [local, remote] of Object.entries(wanted)) {
        const target = path.join(dir, local)
        if (fs.existsSync(target) && fs.statSync(target).size > 1024) {
            console.log(`  ${local} (cached)`)
            continue
        }
        console.log(`  ${local}${local === remote ? "" : ` <- ${remote}`}`)
        await download(`https://huggingface.co/${repo}/resolve/${revision}/${remote}`, target)
        console.log(`  ${local} ${(fs.statSync(target).size / 1e6).toFixed(0)} MB`)
    }
    return dir
}

async function main() {
    const args = process.argv.slice(2)
    const ids = args.includes("--all") ? Object.keys(SETS) : args
    if (!ids.length) {
        console.log(`usage: getBenchModels.js <set...> | --all\n\nsets:`)
        for (const [id, repo] of Object.entries(SETS)) console.log(`  ${id.padEnd(14)} ${repo}`)
        process.exit(1)
    }

    for (const id of ids) {
        try {
            await fetchSet(id)
        } catch (err) {
            console.error(`\n${id}: ${err.message}`)
            process.exitCode = 1
        }
    }
    console.log(`\nmodels -> ${benchModelRoot()}`)
}

main()
