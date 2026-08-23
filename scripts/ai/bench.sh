#!/usr/bin/env bash
# AI BENCH - run the matrix one slice per process.
#
# A loaded recognizer holds ~2 GB of native ONNX state, and dropping the JS reference does not tell
# node's GC that 2 GB became free. Running every fixture set against every model set in one vitest
# worker therefore piles the sessions up until the worker is killed - observed at 772s in. One
# process per (fixture set, variant) keeps each run to a single model, which also makes the
# resource numbers attributable.
set -u
cd "$(dirname "$0")/../.."

STAMP=$(date +%s000)
VARIANTS=${VARIANTS:-"batch en-1120|stream en-1120|stream en-160|stream multi-1120|stream multi-320"}
SETS=${SETS:-$(node -e '
const {listManifests, loadFixtureSet, availableFixtures} = require("./build/electron/ai/speech/bench/fixtures.js")
' 2>/dev/null || echo "")}

# discover fixture sets from the manifests on this machine
if [ -z "$SETS" ]; then
    SETS=$(ls src/electron/ai/speech/bench/manifests/*.json 2>/dev/null | xargs -n1 basename | sed 's/\.json$//')
    FR="${FREESHOW_AI_FIXTURES:-$HOME/Library/Application Support/FreeShow/bin/bench/fixtures}"
    SETS="$SETS $(ls "$FR"/*.json 2>/dev/null | xargs -n1 basename 2>/dev/null | sed 's/\.json$//')"
fi

fail=0
for set_file in $SETS; do
    # manifest file name is not the set id; read the id out of the manifest
    for id in $(node -e '
        const fs=require("fs"), path=require("path"), os=require("os")
        const fr = process.env.FREESHOW_AI_FIXTURES || path.join(os.homedir(), "Library/Application Support/FreeShow/bin/bench/fixtures")
        for (const p of [path.join("src/electron/ai/speech/bench/manifests", process.argv[1]+".json"), path.join(fr, process.argv[1]+".json")])
            if (fs.existsSync(p)) { console.log(JSON.parse(fs.readFileSync(p,"utf8")).id); break }
    ' "$set_file"); do
        IFS='|' read -ra V <<< "$VARIANTS"
        for variant in "${V[@]}"; do
            echo "### $id / $variant"
            AI_BENCH=1 AI_BENCH_STAMP=$STAMP AI_BENCH_SET="$id" AI_BENCH_VARIANT="$variant" \
                npx vitest run --config config/testing/vitest.config.ts src/electron/ai/speech/bench/bench.test.ts 2>&1 \
                | grep -E "^  [a-z]|WER|lag|decode|Error|FAIL" | head -20
            [ ${PIPESTATUS[0]} -ne 0 ] && fail=$((fail+1))
        done
    done
done
echo "### done ($fail failed slices). Reports: test-output/ai-bench/"
