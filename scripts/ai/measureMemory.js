#!/usr/bin/env node
// AI BENCH - peak RSS per model set.
//
// A full matrix in one process killed the vitest worker, which put a number nobody had measured on
// the critical path: a loaded recognizer holds roughly 2 GB of native ONNX state for a 650 MB int8
// model. On an 8 GB machine also running FreeShow's renderer, its output windows and video decode,
// that is a product risk in its own right - independent of latency or accuracy.
//
// Measured in a CHILD PROCESS per set, because native allocations are not returned to the OS
// promptly and a second model loaded in the same process would report the first one's high-water
// mark. Run it when nothing else is competing for memory.
//
//   node scripts/ai/measureMemory.js [set...]

const { execFileSync } = require("child_process")
const fs = require("fs")
const os = require("os")
const path = require("path")

function benchModelRoot() {
    const home = os.homedir()
    const base =
        process.platform === "darwin"
            ? path.join(home, "Library", "Application Support", "FreeShow")
            : process.platform === "win32"
              ? path.join(process.env.APPDATA || path.join(home, "AppData", "Roaming"), "FreeShow")
              : path.join(process.env.XDG_CONFIG_HOME || path.join(home, ".config"), "FreeShow")
    return { bench: path.join(base, "bin", "bench", "models"), app: path.join(base, "bin", "nemotron", "models") }
}

const CHILD = `
const path = require("path")
const sherpa = require("sherpa-onnx-node")
const dir = process.argv[2]
const before = process.memoryUsage().rss
const recognizer = new sherpa.OnlineRecognizer({
    featConfig: { sampleRate: 16000, featureDim: 128 },
    modelConfig: {
        transducer: { encoder: path.join(dir, "encoder.int8.onnx"), decoder: path.join(dir, "decoder.int8.onnx"), joiner: path.join(dir, "joiner.int8.onnx") },
        tokens: path.join(dir, "tokens.txt"), numThreads: 2, provider: "cpu", debug: 0
    },
    decodingMethod: "greedy_search", enableEndpoint: false
})
// decode a little silence: the session allocates its working buffers lazily, so loading alone
// understates what a live session actually holds
const stream = recognizer.createStream()
stream.acceptWaveform({ sampleRate: 16000, samples: new Float32Array(16000 * 5) })
while (recognizer.isReady(stream)) recognizer.decode(stream)
console.log(JSON.stringify({ beforeMb: before / 1e6, afterMb: process.memoryUsage().rss / 1e6 }))
`

function measure(label, dir) {
    if (!fs.existsSync(path.join(dir, "encoder.int8.onnx"))) return null

    const script = path.join(os.tmpdir(), `freeshow-mem-${process.pid}.js`)
    fs.writeFileSync(script, CHILD)
    try {
        // /usr/bin/time -l reports the true high-water mark; process.memoryUsage() inside the child
        // misses native allocations already returned to the allocator but not to the OS
        const out = execFileSync("/usr/bin/time", ["-l", process.execPath, script, dir], {
            encoding: "utf8",
            stdio: ["ignore", "pipe", "pipe"],
            env: { ...process.env, NODE_PATH: path.join(__dirname, "..", "..", "node_modules") }
        })
        return { label, json: JSON.parse(out.trim().split("\n")[0]) }
    } catch (err) {
        const stderr = String(err.stderr || "")
        const peak = stderr.match(/(\d+)\s+maximum resident set size/)
        const stdout = String(err.stdout || "").trim().split("\n")[0]
        return { label, json: stdout ? JSON.parse(stdout) : null, peakMb: peak ? Number(peak[1]) / 1e6 : null }
    } finally {
        fs.unlinkSync(script)
    }
}

function main() {
    const roots = benchModelRoot()
    const wanted = process.argv.slice(2)
    const sets = [{ label: "en-1120 (shipped)", dir: roots.app }]
    if (fs.existsSync(roots.bench)) for (const id of fs.readdirSync(roots.bench)) sets.push({ label: id, dir: path.join(roots.bench, id) })

    console.log(`\n${"model set".padEnd(20)}${"peak RSS".padStart(12)}`)
    console.log("-".repeat(32))
    for (const set of sets) {
        if (wanted.length && !wanted.includes(set.label)) continue
        const result = measure(set.label, set.dir)
        if (!result) continue
        const mb = result.peakMb ?? result.json?.afterMb
        console.log(`${set.label.padEnd(20)}${(mb ? `${(mb / 1000).toFixed(2)} GB` : "-").padStart(12)}`)
    }
    console.log(`\nMeasured in a child process per set: native allocations are not returned to the OS`)
    console.log(`promptly, so a second model in the same process reports the first one's high-water mark.`)
}

main()
