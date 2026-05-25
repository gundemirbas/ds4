#!/usr/bin/env bash
# Captured-decode determinism regression backstop.
#
# Runs the same DS4 greedy generation N times under both
# DS4_CUDA_LAYER_GRAPHS=0 (eager) and =1 (per-layer CUDA-graph capture)
# and compares the generated token-id dumps.  PASS = all OFF runs equal,
# all ON runs equal, and OFF == ON (the parity gate).
#
# It compares the SELECTED token-id sequence from the --dump-logprobs
# JSON -- not the console, and not the whole JSON file.  The console
# carries mode-dependent perf lines (capture vs eager run at different
# tok/s), so a console MD5 fakes false FAILs.  The JSON also carries
# per-token logit/logprob floats (%.9g): those drift by FP-tiny
# reduction-order noise that is NOT a decode divergence, so a whole-file
# MD5 fakes false FAILs too.  The greedy argmax token-id sequence is the
# authoritative per-token decode signal.
#
# Step 7 closed this gate: captured decode is bit-identical to eager
# through n=256 on sm_120 (PRO 6000) and sm_121 (GB10).  This script is
# the permanent regression backstop -- run it after any change to the
# decode path, the layer-graph cache, or the scalar substrates.
#
# Why the default is n=256, not a handful of tokens: captured-graph state
# bugs onset at staggered sequence positions, so a short run misses them.
# The three Step 7 root causes surfaced at pos 0 (stale decode token on
# the hash-mode router), pos ~25-49 (frozen compressed-row counter on the
# FP8-KV emit), and pos 128 (the sliding-window-attention saturation
# boundary).  A run shorter than ~256 tokens would have passed while
# still broken.  To localize a *new* divergence, pair this with the
# per-kernel hash dump (DS4_CUDA_LAYER_GRAPHS_HASH_DUMP=1; see the comment
# block above the implementation in ds4_cuda.cu).
#
# Long-context profile (added 2026-05-26 for Opp C Phase 1A.4):
#   At long input context, baseline-eager decode is currently
#   non-deterministic past roughly 32-64 decoded tokens -- 3x baseline-
#   eager runs at long-prompt + n=128 produce 3 distinct token-id MD5s,
#   while the same 3 runs truncated to the first 32 tokens match. This
#   is unrelated to layer-graph capture or to the FP8 mirror (FP8 OFF +
#   capture OFF reproduces it), and is tracked separately (see
#   local/docs/ds4_long_context_nondeterminism_2026-05-26.md). Until
#   that pre-existing noise source is fixed, the long-context gate is
#   bounded at n=32 -- which still exercises the indexer-fires regime,
#   thousands of compressed rows per layer, and (in FP8-on runs) tens of
#   thousands of FP8 read-path blocks. See PROFILE below.
#
# Usage:  bash tests/cuda_layer_graph_determinism_probe.sh [N] [PROMPT] [NTOK]
#         N      -- run count per mode.       Default 3.
#         PROMPT -- one-shot prompt.          Default a short explainer.
#                   Prefix with '@' to read from a file, e.g.
#                   "@tests/long_context_story_prompt.txt".
#         NTOK   -- generated tokens per run. Default 256 (do not lower
#                   below 256 for the SHORT-prompt gate -- see above).
#
# Recommended profiles:
#   short (the default):
#     bash tests/cuda_layer_graph_determinism_probe.sh
#   long-context, deterministic regime (n=32):
#     bash tests/cuda_layer_graph_determinism_probe.sh \
#         3 "@tests/long_context_story_prompt.txt" 32
#   long-context past n>=64 is currently KNOWN-FLAKY in baseline (not a
#   gate); run it only to characterize the pre-existing noise.
#
# Repeated-batch tip: each ds4 invocation reloads the ~87 GB model
# (~40 s).  For large N, bring up ds4_weight_server once (VMM upload,
# scope=base) and point clients at it via DS4_CUDA_WEIGHT_IPC_MANIFEST --
# per-invocation startup drops to ~14 s.  Shut the server down afterward.
#
# Run from the ds4 source root after building `make cuda CUDA_ARCH=sm_120`.

set -uo pipefail

N=${1:-3}
PROMPT=${2:-"Explain how a transformer attention mechanism works in three short paragraphs."}
NTOK=${3:-256}

DS4=./ds4
TMPDIR=${TMPDIR:-/tmp}/ds4_determinism_probe.$$
mkdir -p "$TMPDIR"
trap "rm -rf $TMPDIR" EXIT

# If PROMPT starts with '@', the remainder is a path to a file containing
# the prompt -- routed through ds4's --prompt-file so prompts larger than
# ARG_MAX (the long-context profiles) work.  Otherwise the prompt is
# passed inline via -p.
PROMPT_ARGS=(-p "$PROMPT")
case "$PROMPT" in
    @*)
        PROMPT_FILE="${PROMPT#@}"
        if [ ! -r "$PROMPT_FILE" ]; then
            echo "probe: prompt file not readable: $PROMPT_FILE" >&2
            exit 64
        fi
        PROMPT_ARGS=(--prompt-file "$PROMPT_FILE")
        ;;
esac

run_one() {
    local mode_label="$1"
    local mode_env="$2"
    local i="$3"
    local lp="$TMPDIR/${mode_label}_${i}.json"
    local con="$TMPDIR/${mode_label}_${i}.con"
    pkill -9 -f "$DS4 --cuda" 2>/dev/null || true
    sleep 1
    env $mode_env $DS4 --cuda --temp 0 -n "$NTOK" "${PROMPT_ARGS[@]}" \
        --dump-logprobs "$lp" --logprobs-top-k 1 > "$con" 2>&1
    # MD5 the SELECTED token-id sequence only (see the header for why
    # neither the console nor the whole JSON file is compared).
    local ids
    ids=$(grep -oE '"selected":\{"id":-?[0-9]+' "$lp" 2>/dev/null \
          | grep -oE -- '-?[0-9]+$' | tr '\n' ',')
    if [ -z "$ids" ]; then
        # No tokens dumped (ds4 crashed / wrote no JSON). Emit a
        # run-unique marker so empty runs can never collude into a
        # false PASS.
        echo "EMPTY-${mode_label}-${i}"
    else
        printf '%s' "$ids" | md5sum | awk '{print $1}'
    fi
}

echo "==== DS4_CUDA_LAYER_GRAPHS=0 (baseline), n=$NTOK ===="
declare -a off_md5s
for i in $(seq 1 $N); do
    md5=$(run_one off "DS4_CUDA_LAYER_GRAPHS=0" $i)
    off_md5s+=("$md5")
    echo "  run $i: $md5"
done

echo
echo "==== DS4_CUDA_LAYER_GRAPHS=1 (capture path), n=$NTOK ===="
declare -a on_md5s
for i in $(seq 1 $N); do
    md5=$(run_one on "DS4_CUDA_LAYER_GRAPHS=1" $i)
    on_md5s+=("$md5")
    echo "  run $i: $md5"
done

echo
echo "==== Determinism report ===="

off_unique=$(printf '%s\n' "${off_md5s[@]}" | sort -u | wc -l)
on_unique=$(printf '%s\n' "${on_md5s[@]}" | sort -u | wc -l)

echo "  OFF unique MD5s: $off_unique / $N"
echo "  ON  unique MD5s: $on_unique / $N"

if [ "$off_unique" -eq 1 ] && [ "$on_unique" -eq 1 ] && [ "${off_md5s[0]}" = "${on_md5s[0]}" ]; then
    echo "  RESULT: PASS -- OFF and ON both deterministic and identical"
    exit 0
elif [ "$off_unique" -eq 1 ] && [ "$on_unique" -eq 1 ]; then
    echo "  RESULT: FAIL/divergence -- OFF and ON each deterministic but differ"
    echo "    OFF: ${off_md5s[0]}"
    echo "    ON:  ${on_md5s[0]}"
    exit 1
elif [ "$off_unique" -eq 1 ] && [ "$on_unique" -gt 1 ]; then
    echo "  RESULT: FAIL/non-determinism -- OFF deterministic, ON varies across runs"
    echo "    OFF: ${off_md5s[0]}"
    echo "    ON values:"
    for m in "${on_md5s[@]}"; do echo "      $m"; done
    exit 2
elif [ "$off_unique" -gt 1 ] && [ "$on_unique" -eq 1 ]; then
    # Captured graphs deterministic, eager is not.  At long context this
    # is the documented pre-existing baseline-eager noise (see the
    # "Long-context profile" comment block at the top of this file and
    # local/docs/ds4_long_context_nondeterminism_2026-05-26.md): the
    # noise is not introduced by capture, so OFF varying while ON is
    # stable is a baseline-noise signal, not a capture regression.
    # Treated as a non-zero exit so the operator notices, but with a
    # distinct code -- callers running this profile knowingly can ignore
    # exit 4 while still failing on the other modes.
    echo "  RESULT: BASELINE-NOISE -- OFF varies, ON deterministic (capture is NOT the regression source)"
    echo "    OFF values:"
    for m in "${off_md5s[@]}"; do echo "      $m"; done
    echo "    ON:  ${on_md5s[0]}"
    exit 4
else
    # Both modes varying.  Captured replay reuses one captured exec, so
    # any noise that appears under ON also appears under OFF; the
    # converse is also possible.  Treat as "noise of unknown source --
    # bisect with the per-kernel hash dump."
    echo "  RESULT: UNEXPECTED -- both modes show variance"
    echo "    OFF values:"
    for m in "${off_md5s[@]}"; do echo "      $m"; done
    echo "    ON values:"
    for m in "${on_md5s[@]}"; do echo "      $m"; done
    exit 3
fi
