export interface EngineStatus {
    /** Files are present but not the pinned revision - a re-download is what fixes it, not a first download. */
    outdated?: boolean
    ready: boolean
    localPath?: string | null
    downloadedModels?: string[]
    supported?: boolean // engines with a native addon (nemotron) report whether this platform can run it at all
}

export interface AiSetupOptions {
    action: "download" | "cancel" | "delete" | "verify"
    engineId: string
    modelId?: string
    customPath?: string
}

// the user's provider/model pick - armed once, then any feature may run completions against it
export interface LlmSelection {
    provider: string
    model: string
}

export interface LLMCompletionOptions {
    systemPrompt?: string
    prompt: string
    jsonSchema?: any
    temperature?: number
    maxTokens?: number
    signal?: AbortSignal
}
