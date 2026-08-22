// AI BENCH - shared summary statistics. Its own file so pacer/metrics can both use it without a
// circular import (pacer -> runner -> metrics -> pacer).

export interface Distribution {
    n: number
    mean: number
    p50: number
    p90: number
    p95: number
    p99: number
    max: number
}

export function percentile(sorted: number[], p: number): number {
    if (!sorted.length) return 0
    const index = Math.min(sorted.length - 1, Math.max(0, Math.ceil((p / 100) * sorted.length) - 1))
    return sorted[index]
}

export function describe(values: number[]): Distribution {
    if (!values.length) return { n: 0, mean: 0, p50: 0, p90: 0, p95: 0, p99: 0, max: 0 }

    const sorted = [...values].sort((a, b) => a - b)
    return {
        n: sorted.length,
        mean: sorted.reduce((sum, value) => sum + value, 0) / sorted.length,
        p50: percentile(sorted, 50),
        p90: percentile(sorted, 90),
        p95: percentile(sorted, 95),
        p99: percentile(sorted, 99),
        max: sorted[sorted.length - 1]
    }
}

/**
 * Bootstrap percentile confidence interval. Fixture sets are small (tens of clips), and a 3-point
 * recall difference between two engines on 30 clips is noise - printing an interval next to every
 * headline number is what stops a decision being made on it.
 *
 * `seed` keeps it deterministic: the same results must produce the same report, or two runs of
 * the reporter disagree and nobody trusts either.
 */
export function bootstrapCi(values: number[], statistic: (sample: number[]) => number, iterations = 2000, seed = 0x5eed): { low: number; high: number } {
    if (values.length < 2) return { low: NaN, high: NaN }

    // xorshift32 - deterministic, and good enough for resampling indices
    let state = seed || 1
    const nextIndex = (limit: number) => {
        state ^= state << 13
        state ^= state >>> 17
        state ^= state << 5
        return Math.abs(state) % limit
    }

    const estimates: number[] = []
    for (let iteration = 0; iteration < iterations; iteration++) {
        const sample = new Array<number>(values.length)
        for (let i = 0; i < values.length; i++) sample[i] = values[nextIndex(values.length)]
        estimates.push(statistic(sample))
    }
    estimates.sort((a, b) => a - b)

    return { low: percentile(estimates, 2.5), high: percentile(estimates, 97.5) }
}

export const mean = (values: number[]) => (values.length ? values.reduce((sum, value) => sum + value, 0) / values.length : 0)
