#!/bin/bash
#SBATCH --job-name=glm5_api
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --account=project_462001047
#SBATCH --output=glm5_api_%j.out

# Experiment 5 — API mode, SWE-bench tasks via Singularity.
# Runs OUTSIDE the LAIF container so singularity exec works without nesting.
# Requires openai in myvenv — install once with:
#   singularity exec $LAIF_SIF pip install --target $MYVENV/lib/python3.12/site-packages "openai>=1.0.0"
#
# Submit with:
#   HF_TOKEN=hf_... sbatch run_agent_api.sh

module load LUMI/25.03
module use /appl/local/laifs/modules
module load lumi-aif-singularity-bindings   # makes /scratch visible inside singularity

MYVENV=/scratch/project_462001047/stockmj/myvenv
SIF_DIR=/scratch/project_462001047/stockmj/sif_images
SCRIPT_DIR=/scratch/project_462001047/stockmj/lumi_glm_test_5

# Run agent directly (no LAIF wrapper) — singularity exec is called per agent step
srun env \
    PYTHONPATH="$MYVENV/lib/python3.12/site-packages" \
    HF_TOKEN="$HF_TOKEN" \
    python3 -u $SCRIPT_DIR/test_agent.py \
        --tasks $SCRIPT_DIR/tasks_swe.jsonl \
        --work-dir $SCRIPT_DIR/runs \
        --model zai-org/GLM-4.7-Flash \
        --max-new-tokens 1000 \
        --step-limit 15 \
        --use-api \
        --singularity-sif-dir "$SIF_DIR"
