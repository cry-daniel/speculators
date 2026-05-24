#!/usr/bin/env bash
# Print or run SpecLink experiment matrices.

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

ROOT=""
MATRIX="baseline"
DRY_RUN="true"
DATASET="data/math_reasoning.jsonl"
MAX_TOKENS="512"
ACCURACY_LIMIT=""
BENCHMARK_LIMIT=""
RATES="1,2,4"
REPEATS="0,1,2"
K_VALUES="2,4,8"
SPECLINK_LAYOUTS="independent_topk,snapkv_static,shared_only,speclink_fixed,speclink_prob,speclink_prob_fallback"
SPECLINK_DRAFT_METHODS="eagle3,peagle"
SPECLINK_BLOCK_SIZES="32"
SPECLINK_SHARED_BUDGETS="16,32,64"
SPECLINK_PRIVATE_MIN_VALUES="0"
SPECLINK_PRIVATE_MAX_VALUES="8,16"
SPECLINK_LAMBDA_RISK_VALUES="0,1"
SPECLINK_FALLBACK_VALUES="0,1"
SPECLINK_ALPHA_VALUES="8"
SPECLINK_BETA_VALUES="8"
SPECLINK_TOPK_PER_TOKEN="32"
SPECLINK_RISK_THRESHOLD="0.35"
SKIP_COMPLETE="false"
NEXT_PORT=9000

usage() {
    cat << EOF
Usage: $0 --root DIR [--matrix baseline|breakdown|speclink-plan|speclink-g2|speclink-serving] [--run]

Defaults print commands only. Use --run to execute them.

Options:
  --root DIR              Result root, e.g. results/speclink_TIMESTAMP
  --matrix NAME           baseline, breakdown, speclink-plan, speclink-g2, or speclink-serving
  --dataset PATH          Dataset JSONL (default: data/math_reasoning.jsonl)
  --max-tokens N          Max generated tokens (default: 512)
  --accuracy-limit N      Accuracy limit; omit for full dataset
  --benchmark-limit N     GuideLLM subset limit; omit for full dataset
  --rates CSV             GuideLLM rates (default: 1,2,4)
  --repeats CSV           Repeat ids (default: 0,1,2)
  --k-values CSV          Speculative lengths for non-dense methods (default: 2,4,8)
  --speclink-layouts CSV  Layouts for speclink-plan matrix
  --speclink-draft-methods CSV
                          Draft families for speclink-g2/speclink-serving (default: eagle3,peagle)
  --block-sizes CSV       SPECLINK_BLOCK_SIZE values for speclink-g2
  --shared-budgets CSV    SPECLINK_SHARED_BUDGET values for speclink-g2
  --private-min-values CSV
                          SPECLINK_PRIVATE_MIN values for speclink-g2
  --private-max-values CSV
                          SPECLINK_PRIVATE_MAX values for speclink-g2
  --lambda-risk-values CSV
                          SPECLINK_LAMBDA_RISK values for speclink-g2
  --fallback-values CSV   0 or 1; maps to speclink_prob or speclink_prob_fallback
  --alpha-values CSV      SPECLINK_ALPHA values for speclink-g2
  --beta-values CSV       SPECLINK_BETA values for speclink-g2
  --topk-per-token N      SPECLINK_TOPK_PER_TOKEN for speclink-g2
  --risk-threshold N      SPECLINK_RISK_THRESHOLD for speclink-g2
  --skip-complete         Skip rows whose required output files already exist
  --run                   Execute commands instead of printing
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --root) ROOT="$2"; shift 2 ;;
        --matrix) MATRIX="$2"; shift 2 ;;
        --dataset) DATASET="$2"; shift 2 ;;
        --max-tokens) MAX_TOKENS="$2"; shift 2 ;;
        --accuracy-limit) ACCURACY_LIMIT="$2"; shift 2 ;;
        --benchmark-limit) BENCHMARK_LIMIT="$2"; shift 2 ;;
        --rates) RATES="$2"; shift 2 ;;
        --repeats) REPEATS="$2"; shift 2 ;;
        --k-values) K_VALUES="$2"; shift 2 ;;
        --speclink-layouts) SPECLINK_LAYOUTS="$2"; shift 2 ;;
        --speclink-draft-methods) SPECLINK_DRAFT_METHODS="$2"; shift 2 ;;
        --block-sizes) SPECLINK_BLOCK_SIZES="$2"; shift 2 ;;
        --shared-budgets) SPECLINK_SHARED_BUDGETS="$2"; shift 2 ;;
        --private-min-values) SPECLINK_PRIVATE_MIN_VALUES="$2"; shift 2 ;;
        --private-max-values) SPECLINK_PRIVATE_MAX_VALUES="$2"; shift 2 ;;
        --lambda-risk-values) SPECLINK_LAMBDA_RISK_VALUES="$2"; shift 2 ;;
        --fallback-values) SPECLINK_FALLBACK_VALUES="$2"; shift 2 ;;
        --alpha-values) SPECLINK_ALPHA_VALUES="$2"; shift 2 ;;
        --beta-values) SPECLINK_BETA_VALUES="$2"; shift 2 ;;
        --topk-per-token) SPECLINK_TOPK_PER_TOKEN="$2"; shift 2 ;;
        --risk-threshold) SPECLINK_RISK_THRESHOLD="$2"; shift 2 ;;
        --skip-complete) SKIP_COMPLETE="true"; shift ;;
        --run) DRY_RUN="false"; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "[ERROR] unknown option: $1" >&2; usage; exit 1 ;;
    esac
done

if [[ -z "${ROOT}" ]]; then
    echo "[ERROR] --root is required" >&2
    usage
    exit 1
fi

csv_items() {
    local value="$1"
    value="${value//,/ }"
    printf '%s\n' ${value}
}

quote_cmd() {
    printf '%q ' "$@"
    printf '\n'
}

slug() {
    local value="$1"
    value="${value//./p}"
    value="${value//- /m}"
    value="${value//-/m}"
    printf '%s' "${value}"
}

run_or_print() {
    if [[ "${DRY_RUN}" == "true" ]]; then
        quote_cmd "$@"
    else
        "$@"
    fi
}

is_complete_run() {
    local method="$1"
    local out_dir="$2"
    local required=(
        "${out_dir}/guidellm_results.json"
        "${out_dir}/accuracy_summary.json"
        "${out_dir}/env.txt"
        "${out_dir}/command.txt"
    )
    if [[ "${method}" != "dense" ]]; then
        required+=("${out_dir}/acceptance_analysis.txt")
    fi
    if [[ "${MATRIX}" == "breakdown" || "${MATRIX}" == "speclink-plan" || "${MATRIX}" == "speclink-g2" || "${MATRIX}" == "speclink-serving" ]]; then
        required+=("${out_dir}/profile_events.jsonl")
    fi
    if [[ "${MATRIX}" == "speclink-plan" || "${MATRIX}" == "speclink-g2" || "${MATRIX}" == "speclink-serving" ]]; then
        required+=("${out_dir}/live_sparse_trace.jsonl")
    fi
    [[ ! -f "${out_dir}/failures.md" ]] || return 1
    local path
    for path in "${required[@]}"; do
        [[ -f "${path}" ]] || return 1
    done
    local expected_accuracy_limit="${ACCURACY_LIMIT:-full}"
    local expected_benchmark_limit="${BENCHMARK_LIMIT:-full}"
    grep -qx "accuracy_limit=${expected_accuracy_limit}" "${out_dir}/env.txt" || return 1
    grep -qx "benchmark_limit=${expected_benchmark_limit}" "${out_dir}/env.txt" || return 1
    python - "${out_dir}/guidellm_results.json" "${out_dir}/accuracy_summary.json" <<'PY'
import json
import sys

guidellm_path = sys.argv[1]
accuracy_path = sys.argv[2]
with open(accuracy_path, encoding="utf-8") as handle:
    summary = json.load(handle)
if int(summary.get("errors", 0)) != 0:
    sys.exit(1)
with open(guidellm_path, encoding="utf-8") as handle:
    guidellm = json.load(handle)

def find_nested(obj, key):
    if isinstance(obj, dict):
        if key in obj:
            return obj[key]
        for value in obj.values():
            found = find_nested(value, key)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for value in obj:
            found = find_nested(value, key)
            if found is not None:
                return found
    return None

def request_total(value):
    if isinstance(value, dict):
        for key in ("successful", "total"):
            if isinstance(value.get(key), int):
                return int(value[key])
    if isinstance(value, int):
        return value
    return None

benchmark = (guidellm.get("benchmarks") or [{}])[0]
metrics = benchmark.get("metrics", {})
n_requests = request_total(find_nested(metrics, "request_totals"))
accuracy_n = summary.get("n")
sys.exit(0 if n_requests is not None and accuracy_n is not None and int(n_requests) == int(accuracy_n) else 1)
PY
}

common_args() {
    local out=(
        --dataset "${DATASET}"
        --max-tokens "${MAX_TOKENS}"
    )
    [[ -n "${ACCURACY_LIMIT}" ]] && out+=(--accuracy-limit "${ACCURACY_LIMIT}")
    [[ -n "${BENCHMARK_LIMIT}" ]] && out+=(--benchmark-limit "${BENCHMARK_LIMIT}")
    printf '%s\n' "${out[@]}"
}

run_method() {
    local method="$1"
    local k="$2"
    local rate="$3"
    local repeat="$4"
    local out_dir="$5"
    local port="$6"
    shift 6
    local extra_env=("$@")
    local args
    mapfile -t args < <(common_args)
    if [[ "${SKIP_COMPLETE}" == "true" ]] && is_complete_run "${method}" "${out_dir}"; then
        echo "[INFO] Skipping complete row: ${out_dir}" >&2
        return
    fi
    local cmd=(
        env
        "GUIDELLM_RATE=${rate}"
        "REQUEST_TYPE=chat_completions"
    )
    cmd+=("${extra_env[@]}")
    cmd+=(
        conda run -n spec bash "${SCRIPT_DIR}/speclink_run_method.sh"
        --method "${method}"
        --num-spec-tokens "${k}"
        --guidellm-rate "${rate}"
        --request-type chat_completions
        --port "${port}"
        --output-dir "${out_dir}"
        --repeat-id "${repeat}"
    )
    cmd+=("${args[@]}")
    if [[ "${method}" != "dense" ]]; then
        if [[ "${MATRIX}" == "baseline" ]]; then
            cmd+=(--dense-reference-jsonl "${ROOT}/02_baselines/dense_rate${rate}_r${repeat}/accuracy_outputs.jsonl")
        else
            local dense_ref="${ROOT}/02_baselines/dense_rate${rate}_r0/accuracy_outputs.jsonl"
            [[ -f "${dense_ref}" ]] || dense_ref="${ROOT}/02_baselines/dense_smoke/accuracy_outputs.jsonl"
            cmd+=(--dense-reference-jsonl "${dense_ref}")
        fi
    fi
    run_or_print "${cmd[@]}"
}

method_offset() {
    case "$1" in
        dense) echo 0 ;;
        eagle3) echo 100 ;;
    peagle) echo 200 ;;
    speclink) echo 300 ;;
        *) echo 900 ;;
    esac
}

case "${MATRIX}" in
    baseline)
        for repeat in $(csv_items "${REPEATS}"); do
            for rate in $(csv_items "${RATES}"); do
                run_method dense 0 "${rate}" "${repeat}" \
                    "${ROOT}/02_baselines/dense_rate${rate}_r${repeat}" \
                    "$((8000 + repeat * 1000 + rate))"
                for method in eagle3 peagle; do
                    for k in $(csv_items "${K_VALUES}"); do
                        run_method "${method}" "${k}" "${rate}" "${repeat}" \
                            "${ROOT}/02_baselines/${method}_k${k}_rate${rate}_r${repeat}" \
                            "$((8100 + repeat * 1000 + $(method_offset "${method}") + rate * 10 + k))"
                    done
                done
            done
        done
        ;;
    breakdown)
        for method in eagle3 peagle; do
            for k in $(csv_items "${K_VALUES}"); do
                for rate in 1 4; do
                    out_dir="${ROOT}/03_breakdown/${method}_k${k}_rate${rate}_profile"
                    run_method "${method}" "${k}" "${rate}" 0 "${out_dir}" "$((8300 + $(method_offset "${method}") + rate * 10 + k))" \
                        "SPECLINK_PROFILE=1" \
                        "SPECLINK_METHOD=${method}" \
                        "SPECLINK_PROFILE_OUT=${out_dir}/profile_events.jsonl"
                done
            done
        done
        ;;
    speclink-plan)
        for layout in $(csv_items "${SPECLINK_LAYOUTS}"); do
            for k in $(csv_items "${K_VALUES}"); do
                out_dir="${ROOT}/05_speclink/${layout}_k${k}_rate1_plan"
                run_method speclink "${k}" 1 0 "${out_dir}" "$((8500 + $(method_offset speclink) + k))" \
                    "SPECLINK_ENABLE=1" \
                    "SPECLINK_MODE=plan_only" \
                    "SPECLINK_LAYOUT=${layout}" \
                    "SPECLINK_METHOD=speclink" \
                    "SPECLINK_PROFILE=1" \
                    "SPECLINK_TRACE_OUT=${out_dir}/live_sparse_trace.jsonl" \
                    "SPECLINK_PROFILE_OUT=${out_dir}/profile_events.jsonl"
            done
        done
        ;;
    speclink-g2)
        for draft_method in $(csv_items "${SPECLINK_DRAFT_METHODS}"); do
            case "${draft_method}" in
                eagle3)
                    draft_speculator="${EAGLE3_SPECULATOR_MODEL:-/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/models/qwen3-8b-eagle3-speculator}"
                    parallel_drafting="false"
                    ;;
                peagle)
                    draft_speculator="${PEAGLE_SPECULATOR_MODEL:-/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/models/qwen3-8b-peagle-speculator}"
                    parallel_drafting="true"
                    ;;
                *)
                    echo "[ERROR] unsupported speclink draft method: ${draft_method}" >&2
                    exit 1
                    ;;
            esac
            for k in $(csv_items "${K_VALUES}"); do
                for block_size in $(csv_items "${SPECLINK_BLOCK_SIZES}"); do
                    for shared_budget in $(csv_items "${SPECLINK_SHARED_BUDGETS}"); do
                        for private_min in $(csv_items "${SPECLINK_PRIVATE_MIN_VALUES}"); do
                            for private_max in $(csv_items "${SPECLINK_PRIVATE_MAX_VALUES}"); do
                                for lambda_risk in $(csv_items "${SPECLINK_LAMBDA_RISK_VALUES}"); do
                                    for fallback in $(csv_items "${SPECLINK_FALLBACK_VALUES}"); do
                                        for alpha in $(csv_items "${SPECLINK_ALPHA_VALUES}"); do
                                            for beta in $(csv_items "${SPECLINK_BETA_VALUES}"); do
                                                case "${fallback}" in
                                                    0|false|off) layout="speclink_prob"; fallback_slug="0" ;;
                                                    1|true|on) layout="speclink_prob_fallback"; fallback_slug="1" ;;
                                                    *)
                                                        echo "[ERROR] unsupported fallback value: ${fallback}" >&2
                                                        exit 1
                                                        ;;
                                                esac
                                                lambda_slug="$(slug "${lambda_risk}")"
                                                alpha_slug="$(slug "${alpha}")"
                                                beta_slug="$(slug "${beta}")"
                                                out_dir="${ROOT}/05_speclink_g2/${draft_method}_${layout}_k${k}_bs${block_size}_sb${shared_budget}_pmin${private_min}_pmax${private_max}_lam${lambda_slug}_a${alpha_slug}_b${beta_slug}_fb${fallback_slug}_rate1_plan"
                                                port="${NEXT_PORT}"
                                                NEXT_PORT=$((NEXT_PORT + 1))
                                                run_method speclink "${k}" 1 0 "${out_dir}" "${port}" \
                                                    "SPECLINK_ENABLE=1" \
                                                    "SPECLINK_MODE=plan_only" \
                                                    "SPECLINK_LAYOUT=${layout}" \
                                                    "SPECLINK_METHOD=${draft_method}_speclink" \
                                                    "SPECLINK_SPECULATOR_MODEL=${draft_speculator}" \
                                                    "SPECLINK_PARALLEL_DRAFTING=${parallel_drafting}" \
                                                    "SPECLINK_PROFILE=1" \
                                                    "SPECLINK_BLOCK_SIZE=${block_size}" \
                                                    "SPECLINK_TOPK_PER_TOKEN=${SPECLINK_TOPK_PER_TOKEN}" \
                                                    "SPECLINK_SHARED_BUDGET=${shared_budget}" \
                                                    "SPECLINK_PRIVATE_MIN=${private_min}" \
                                                    "SPECLINK_PRIVATE_MAX=${private_max}" \
                                                    "SPECLINK_ALPHA=${alpha}" \
                                                    "SPECLINK_BETA=${beta}" \
                                                    "SPECLINK_LAMBDA_RISK=${lambda_risk}" \
                                                    "SPECLINK_RISK_THRESHOLD=${SPECLINK_RISK_THRESHOLD}" \
                                                    "SPECLINK_TRACE_OUT=${out_dir}/live_sparse_trace.jsonl" \
                                                    "SPECLINK_PROFILE_OUT=${out_dir}/profile_events.jsonl"
                                            done
                                        done
                                    done
                                done
                            done
                        done
                    done
                done
            done
        done
        ;;
    speclink-serving)
        layout="speclink_prob"
        block_size="32"
        shared_budget="16"
        private_min="0"
        private_max="16"
        lambda_risk="0"
        alpha="8"
        beta="8"
        fallback_slug="0"
        for draft_method in $(csv_items "${SPECLINK_DRAFT_METHODS}"); do
            case "${draft_method}" in
                eagle3)
                    draft_speculator="${EAGLE3_SPECULATOR_MODEL:-/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/models/qwen3-8b-eagle3-speculator}"
                    parallel_drafting="false"
                    ;;
                peagle)
                    draft_speculator="${PEAGLE_SPECULATOR_MODEL:-/ACALAB/stu1/chenruiyang/Code/LLM/SpecLink/models/qwen3-8b-peagle-speculator}"
                    parallel_drafting="true"
                    ;;
                *)
                    echo "[ERROR] unsupported speclink draft method: ${draft_method}" >&2
                    exit 1
                    ;;
            esac
            for k in $(csv_items "${K_VALUES}"); do
                for rate in $(csv_items "${RATES}"); do
                    for repeat in $(csv_items "${REPEATS}"); do
                        lambda_slug="$(slug "${lambda_risk}")"
                        alpha_slug="$(slug "${alpha}")"
                        beta_slug="$(slug "${beta}")"
                        out_dir="${ROOT}/06_serving_rates/${draft_method}_${layout}_k${k}_bs${block_size}_sb${shared_budget}_pmin${private_min}_pmax${private_max}_lam${lambda_slug}_a${alpha_slug}_b${beta_slug}_fb${fallback_slug}_rate${rate}_r${repeat}_plan"
                        port="${NEXT_PORT}"
                        NEXT_PORT=$((NEXT_PORT + 1))
                        run_method speclink "${k}" "${rate}" "${repeat}" "${out_dir}" "${port}" \
                            "SPECLINK_ENABLE=1" \
                            "SPECLINK_MODE=plan_only" \
                            "SPECLINK_LAYOUT=${layout}" \
                            "SPECLINK_METHOD=${draft_method}_speclink" \
                            "SPECLINK_SPECULATOR_MODEL=${draft_speculator}" \
                            "SPECLINK_PARALLEL_DRAFTING=${parallel_drafting}" \
                            "SPECLINK_PROFILE=1" \
                            "SPECLINK_BLOCK_SIZE=${block_size}" \
                            "SPECLINK_TOPK_PER_TOKEN=${SPECLINK_TOPK_PER_TOKEN}" \
                            "SPECLINK_SHARED_BUDGET=${shared_budget}" \
                            "SPECLINK_PRIVATE_MIN=${private_min}" \
                            "SPECLINK_PRIVATE_MAX=${private_max}" \
                            "SPECLINK_ALPHA=${alpha}" \
                            "SPECLINK_BETA=${beta}" \
                            "SPECLINK_LAMBDA_RISK=${lambda_risk}" \
                            "SPECLINK_RISK_THRESHOLD=${SPECLINK_RISK_THRESHOLD}" \
                            "SPECLINK_TRACE_OUT=${out_dir}/live_sparse_trace.jsonl" \
                            "SPECLINK_PROFILE_OUT=${out_dir}/profile_events.jsonl"
                    done
                done
            done
        done
        ;;
    *)
        echo "[ERROR] unsupported matrix: ${MATRIX}" >&2
        usage
        exit 1
        ;;
esac
