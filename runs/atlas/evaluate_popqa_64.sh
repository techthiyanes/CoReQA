#!/usr/bin/env bash
#SBATCH --partition=alvis
#SBATCH --account=SNIC2022-22-1003
#SBATCH --nodes=1
#SBATCH --time=0-12:00:00
#SBATCH --job-name=evaluate-popqa-64
#SBATCH --error=./logs/evaluate-popqa-64-%J.err.log
#SBATCH --output=./logs/evaluate-popqa-64-%J.out.log
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=A100:4

set -eo pipefail

module load git-lfs/3.2.0
module load Python/3.8.6-GCCcore-10.2.0
module load CUDA/11.3.1

export CUDA_LAUNCH_BLOCKING=1
export USERNAME_DIR=snic2022-22-1003
export WANDB_CACHE_DIR=/mimer/NOBACKUP/groups/${USERNAME_DIR}/OUTPUT/.cache/wandb
export TRANSFORMERS_CACHE=/mimer/NOBACKUP/groups/${USERNAME_DIR}/OUTPUT/.cache/huggingface/transformers
export HF_DATASETS_CACHE=/mimer/NOBACKUP/groups/${USERNAME_DIR}/OUTPUT/.cache/huggingface/datasets

source /mimer/NOBACKUP/groups/${USERNAME_DIR}/APP/qa-retriever/venv/bin/activate
echo "PYTHON PATH:"
which python

git config --global user.name "Mehrdad Farahani"
git config --global user.email "m3hrdadfi@gmail.com"
git config --global credential.helper store

SIZE=base
SAVE_DIR=/mimer/NOBACKUP/groups/${USERNAME_DIR}/APP/qa-retriever/experiments
GENERATION_MAX_LENGTH=128
TARGET_MAX_LENGTH=64
GOLD_SCORE_MODE=ppmean
PRECISION=fp32
READER_MODEL_TYPE=google/t5-${SIZE}-lm-adapt
TEXT_MAXLENGTH=512
PER_GPU_BATCH_SIZE=1
N_CONTEXT=5
RETRIEVER_N_CONTEXT=5
MODEL_PATH=/mimer/NOBACKUP/groups/${USERNAME_DIR}/APP/qa-retriever/data/atlas/models/atlas/${SIZE}
MAIN_PORT=$(shuf -i 15000-16000 -n 1)
INDEX_MODE=flat
FAISS_INDEX_TYPE=pq
FAISS_CODE_SIZE=64
TASK=qa
QA_PROMPT_FORMAT="question: {question} answer: <extra_id_0>"
INDEX_PATH=/mimer/NOBACKUP/groups/${USERNAME_DIR}/APP/qa-retriever/data/atlas/indices/atlas_nq/wiki/${SIZE}
EVAL_FILES="/mimer/NOBACKUP/groups/${USERNAME_DIR}/APP/qa-retriever/data/atlas/popqa_data/test.64-shot.jsonl"
GENERATION_NUM_BEAMS=6
NAME=evaluate-popqa-64-size-${SIZE}-ctx-${N_CONTEXT}


# --faiss_index_type="$FAISS_INDEX_TYPE" \
# --faiss_code_size=$FAISS_CODE_SIZE \

echo "EXPERIMENT: ${NAME}"

# srun echo $PWD
srun python src/atlas_evaluate.py \
    --name="$NAME" \
    --generation_max_length=$GENERATION_MAX_LENGTH \
    --target_maxlength=$TARGET_MAX_LENGTH \
    --gold_score_mode="$GOLD_SCORE_MODE" \
    --precision="$PRECISION" \
    --reader_model_type="$READER_MODEL_TYPE" \
    --text_maxlength=$TEXT_MAXLENGTH \
    --model_path="$MODEL_PATH" \
    --eval_data="$EVAL_FILES" \
    --per_gpu_batch_size=$PER_GPU_BATCH_SIZE \
    --n_context=$N_CONTEXT \
    --retriever_n_context=$RETRIEVER_N_CONTEXT \
    --checkpoint_dir="${SAVE_DIR}/${NAME}/" \
    --main_port=$MAIN_PORT \
    --index_mode="$INDEX_MODE" \
    --task="$TASK" \
    --qa_prompt_format="$QA_PROMPT_FORMAT" \
    --load_index_path="$INDEX_PATH" \
    --generation_num_beams=$GENERATION_NUM_BEAMS \
    --write_results
