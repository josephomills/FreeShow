import { createHash } from "crypto"
import fs from "fs"
import os from "os"
import path from "path"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

// A pinned table of three tiny files, so the real 662 MB one does not have to exist to test the
// logic that guards it.
const CONTENT = { "encoder.int8.onnx": "encoder-bytes", "decoder.int8.onnx": "decoder-bytes", "tokens.txt": "tokens-bytes" }
const sha = (value: string) => createHash("sha256").update(value).digest("hex")

vi.mock("../../setup/models/nemotronFiles", () => ({
    NEMOTRON_MODEL_REVISION: "rev-one",
    NEMOTRON_MODEL_FILES: {
        encoder: { file: "encoder.int8.onnx", sha256: sha(CONTENT["encoder.int8.onnx"]) },
        decoder: { file: "decoder.int8.onnx", sha256: sha(CONTENT["decoder.int8.onnx"]) },
        tokens: { file: "tokens.txt", sha256: sha(CONTENT["tokens.txt"]) }
    }
}))

const { isStampValid, verifyModel } = await import("./integrity")

let dir: string

function write(name: string, body: string) {
    fs.writeFileSync(path.join(dir, name), body)
}

beforeEach(() => {
    dir = fs.mkdtempSync(path.join(os.tmpdir(), "freeshow-integrity-"))
    for (const [name, body] of Object.entries(CONTENT)) write(name, body)
})

afterEach(() => fs.rmSync(dir, { recursive: true, force: true }))

describe("nemotron model integrity", () => {
    it("accepts the pinned bytes and stamps them", async () => {
        expect(await verifyModel(dir)).toBe("ok")
        expect(fs.existsSync(path.join(dir, "model.json"))).toBe(true)
    })

    it("reports a superseded model as outdated, not missing", async () => {
        // the distinction is the whole point: "missing" tells a user to download, "outdated" tells
        // them to download AGAIN, and a superseded model still transcribes so nothing looks wrong
        write("encoder.int8.onnx", "bytes from an older pinned revision")
        expect(await verifyModel(dir)).toBe("outdated")
    })

    it("reports an absent file as missing", async () => {
        fs.unlinkSync(path.join(dir, "decoder.int8.onnx"))
        expect(await verifyModel(dir)).toBe("missing")
    })

    it("never stamps a model it rejected", async () => {
        write("tokens.txt", "wrong")
        await verifyModel(dir)
        expect(fs.existsSync(path.join(dir, "model.json"))).toBe(false)
    })

    it("trusts the stamp on the next call without re-reading the files", async () => {
        await verifyModel(dir)
        expect(isStampValid(dir)).toBe(true)

        // hashing 662 MB costs ~1.7s, so a session start must not pay it twice. Prove the cheap
        // path is genuinely cheap by making a read throw.
        const readStream = vi.spyOn(fs, "createReadStream").mockImplementation(() => {
            throw new Error("integrity re-hashed a model it had already stamped")
        })
        expect(await verifyModel(dir)).toBe("ok")
        readStream.mockRestore()
    })

    it("distrusts the stamp when a file changed underneath it", async () => {
        await verifyModel(dir)
        expect(isStampValid(dir)).toBe(true)

        // same name, different bytes - the size/mtime record is what catches this without hashing
        write("encoder.int8.onnx", "swapped after verification, same file name")
        expect(isStampValid(dir)).toBe(false)
        expect(await verifyModel(dir)).toBe("outdated")
    })

    it("distrusts a stamp written for a different pinned revision", async () => {
        await verifyModel(dir)
        const stamp = JSON.parse(fs.readFileSync(path.join(dir, "model.json"), "utf8"))
        stamp.revision = "rev-zero"
        fs.writeFileSync(path.join(dir, "model.json"), JSON.stringify(stamp))

        expect(isStampValid(dir)).toBe(false)
    })

    it("has no stamp before anything is verified", () => {
        expect(isStampValid(dir)).toBe(false)
    })

    it("survives a corrupt stamp rather than throwing", async () => {
        fs.writeFileSync(path.join(dir, "model.json"), "{ this is not json")
        expect(isStampValid(dir)).toBe(false)
        expect(await verifyModel(dir)).toBe("ok")
    })

    it("stamps from digests the downloader already computed", async () => {
        // the download loop hashes every file as it lands; reading 662 MB back to learn the same
        // thing would double the wait at the end of a download
        const readStream = vi.spyOn(fs, "createReadStream").mockImplementation(() => {
            throw new Error("integrity re-hashed files the downloader had already hashed")
        })
        const digests = Object.fromEntries(Object.entries(CONTENT).map(([name, body]) => [name, sha(body)]))

        expect(await verifyModel(dir, digests)).toBe("ok")
        readStream.mockRestore()
        expect(isStampValid(dir)).toBe(true)
    })

    it("rejects downloader digests that do not match the pin", async () => {
        expect(await verifyModel(dir, { "encoder.int8.onnx": sha("something else") })).toBe("outdated")
    })
})
