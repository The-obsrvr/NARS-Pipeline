#!/bin/bash
set -e

# ── Configuration ─────────────────────────────────────────────────────────────
VERSION="v1"
OUT="outputs/${VERSION}"
MODEL="gpt-oss:20b"
TRAIN_QUALITY_MODEL=true

mkdir -p "${OUT}"
echo "Pipeline ${VERSION} → ${OUT}"

export OLLAMA_KEEP_ALIVE=1h

# ── Start three Ollama servers ────────────────────────────────────────────────
OLLAMA_HOST=http://127.0.0.1:11434 ollama serve &

until curl -s http://127.0.0.1:11434/v1/models > /dev/null; do sleep 1; done

# check if model is already pulled
if ! ollama list | grep -q "${MODEL}"; then
    OLLAMA_HOST=http://127.0.0.1:11434 ollama pull "${MODEL}"
else
    echo "${MODEL} already available — skipping pull"
fi

# Warm up all three servers
echo "Warming up models..."
curl -s http://127.0.0.1:11434/api/chat -d "{\"model\":\"${MODEL}\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"stream\":false}" > /dev/null &
WP0=$!

wait $WP0
echo "All three models warmed up."

# ── Steps 1: EDU ───────────────────────────────────
python src/edu_extractor.py \
    --input    Data/samples.jsonl \
    --output   ${OUT}/step1_edu_${VERSION}.jsonl \
    --log-file "${OUT}/log_edu_${VERSION}.log"
echo "Step 1 EDU Extraction complete."

# ── Step 2: PAC ───────────────────────────────────
python src/pac_selector.py \
    --input    ${OUT}/step1_edu_${VERSION}.jsonl \
    --output   ${OUT}/step2_pac_${VERSION}.jsonl \
    --log-file "${OUT}/log_pac_${VERSION}.log"
echo "Step 2 PAC Selection complete."

# ── Step 3: LLM Reasoning ───────────────────────────────────
python src/llm_reasoner.py \
        --input           ${OUT}/step2_pac_${VERSION}.jsonl \
        --output          ${OUT}/step3_reason_${VERSION}.jsonl \
        --ollama-url      http://127.0.0.1:11434/api/chat \
        --model           "${MODEL}" \
        --thinking-budget medium \
        --log-file        "${OUT}/log_rsn_${VERSION}.log" &
PID0=$!

wait $PID0 || { echo "ERROR: reasoning failed (PID=$PID0)"; exit 1; }
echo "Step 3 LLM Reasoning complete."

# ── Step 4: BAS Assembly ──────────────────────────────────────────────────────
python src/bas_assembler.py \
    --input            "${OUT}/step3_reason_${VERSION}.jsonl" \
    --output-repair    "${OUT}/step4_bas_repair_${VERSION}.jsonl" \
    --output-no-repair "${OUT}/step4_bas_no_repair_${VERSION}.jsonl" \
    --ollama-url       http://127.0.0.1:11434/api/chat \
    --model            "${MODEL}" \
    --log-file         "${OUT}/log_bas_${VERSION}.log"

echo "Step 4 BAS Construction complete."

# ── Step 5a: Finetuning Quality Model (OPTIONAL) ──────────────────────────────────────────

if [ "${TRAIN_QUALITY_MODEL}" = "true" ]; then
    echo "Step 5a: Training quality model..."
    python src/train_quality_model.py \
          --args-data  Data/webis-argquality20-full.csv \
          --topic-data Data/webis-argquality20-topics.csv \
          --output     ./quality_model
    echo "Step 5a Finetuning Quality Model complete."
else
    echo "Step 5a skipped (TRAIN_QUALITY_MODEL=false)."
fi

# ── Step 5: Strength Initialization ──────────────────────────────────────────
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# This Step requires Step 5a to be completed.
python src/strength_initializer.py \
    --input-repair      "${OUT}/step4_bas_repair_${VERSION}.jsonl" \
    --input-no-repair   "${OUT}/step4_bas_no_repair_${VERSION}.jsonl" \
    --output-repair     "${OUT}/step5_initialized_repair_AB_${VERSION}.jsonl" \
    --output-no-repair  "${OUT}/step5_initialized_no_repair_AB_${VERSION}.jsonl" \
    --strategy          all \
    --quality-model     ./quality_model \
    --log-file          "${OUT}/log_init_${VERSION}.log"

echo "Step 5 complete."

# ── Step 6: Gradual Semantics ─────────────────────────────────────────────────
python src/gradual_semantics.py \
    --input-repair      "${OUT}/step5_initialized_repair_${VERSION}.jsonl" \
    --input-no-repair   "${OUT}/step5_initialized_no_repair_${VERSION}.jsonl" \
    --output-repair     "${OUT}/step6_semantics_repair_${VERSION}.jsonl" \
    --output-no-repair  "${OUT}/step6_semantics_no_repair_${VERSION}.jsonl" \
    --strategy          all \
    --log-file          "${OUT}/log_gsem_AB_${VERSION}.log"

echo "Step 6 complete."

# ── Step 7: Persuasiveness Detection ─────────────────────────────────────────
python src/persuasiveness_detector.py \
    --input-repair      "${OUT}/step6_semantics_repair_${VERSION}.jsonl" \
    --input-no-repair   "${OUT}/step6_semantics_no_repair_${VERSION}.jsonl" \
    --output-repair     "${OUT}/step7_predictions_repair_${VERSION}.jsonl" \
    --output-no-repair  "${OUT}/step7_predictions_no_repair_${VERSION}.jsonl" \
    --threshold         0.1 0.3 0.5 0.7 0.8 \
    --ground-truth-key  is_delta \
    --log-file          "${OUT}/log_pers_${VERSION}.log"

echo "Step 7 complete."

# ── Evaluation ────────────────────────────────────────────────────────────────
python src/evaluate.py \
    --input-repair      "${OUT}/predictions_repair_${VERSION}.jsonl" \
    --input-no-repair   "${OUT}/predictions_no_repair_${VERSION}.jsonl" \
    --output            "${OUT}/evaluation_results_NARS_${VERSION}.json"

echo "Evaluation complete. Results → ${OUT}/evaluation_results_NARS_${VERSION}.json"
echo "Pipeline complete."


