export interface AiSettings {
    enabled?: boolean
    stt?: SttSettings
    llm?: LlmSettings

    scripture?: AiScriptureSettings
}

interface SttSettings {
    // enabled?: boolean
    micDeviceId?: string

    engine?: string
    // per-engine options
    engineOptions?: {
        [key: string]: SttEngineOptions
    }
}

export interface SttEngineOptions {
    customPath?: string

    language?: string // transcription language
    model?: string
    customModelPath?: string

    /**
     * Nemotron: drive the recognizer as one persistent cache-aware stream instead of re-decoding
     * each utterance on a fresh one. Measured ~3.5x cheaper and ~1.2s lower per-word latency
     * (src/electron/ai/speech/bench/). Off by default for one release - the batch path is what
     * has run in live services so far.
     */
    streamingDecode?: boolean

    // WIP interpretationMode
    interpretationMode?: boolean
    listenLanguage?: string
    spokenLanguages?: string[]
}

interface LlmSettings {
    provider?: string
    model?: string
}

interface AiScriptureSettings {
    enabled?: boolean
    mode?: "confirm" | "auto"
    autoMinConfidence?: number // auto mode only shows detections scoring at/above this percent (default 80 - "high" band); the single gate for references & quotes
    searchBibles?: string[] // legacy tick list - every installed translation is searched now
    mainTranslation?: string // the projection/grounding target; empty = first favourite, else the drawer choice
    displayTranslation?: "drawer" | "matched" // "drawer" = the main translation
    followInDrawer?: boolean // projections also navigate the drawer's scripture view (default on)
    micDeviceId?: string
    provider?: string
    model?: string // legacy single model value (kept as fallback)
    models?: { [key: string]: string }
    customModel?: string
    engine?: string
    whisperModel?: string
    whisperCustomPath?: string
    whisperCustomModelPath?: string
    spokenLanguage?: string
    interpretationMode?: boolean
    listenLanguage?: string
    spokenLanguages?: string[]
    autoCooldownSeconds?: number
    refCooldownSeconds?: number
    maxVerses?: number // caps how many verses of a spoken range project at once (layout stays with the scripture settings)
    voiceCommands?: boolean
}
