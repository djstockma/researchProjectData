#!/bin/bash
#SBATCH --job-name=glm5_gpu4
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=08:00:00
#SBATCH --account=project_462001047
#SBATCH --output=glm5_gpu4_%j.out

# Experiment 5 — scaling comparison: 4 GPUs vs 2 GPUs (run_agent_gpu.sh).
# Same 40 QuixBugs tasks, same model — results go to runs_4gpu/ for side-by-side comparison.
# myvenv has transformers 5.3.0.dev0 (supports glm4_moe_lite).

module load LUMI/25.03
module use /appl/local/laifs/modules
module load lumi-aif-singularity-bindings

LAIF_SIF=/appl/local/laifs/containers/lumi-multitorch-latest.sif
MYVENV=/scratch/project_462001047/stockmj/myvenv
SCRIPT_DIR=/scratch/project_462001047/stockmj/lumi_glm_test_5

export HF_HOME=/scratch/project_462001047/stockmj/hf_cache
export XDG_CACHE_HOME=$HF_HOME

srun singularity exec \
    --bind /scratch:/scratch \
    --env PYTHONPATH="$MYVENV/lib/python3.12/site-packages" \
    --env HF_HOME="$HF_HOME" \
    --env XDG_CACHE_HOME="$HF_HOME" \
    "$LAIF_SIF" \
    python3 -u $SCRIPT_DIR/test_agent.py \
        --tasks $SCRIPT_DIR/tasks.jsonl \
        --work-dir $SCRIPT_DIR/runs/runs_4gpu \
        --model zai-org/GLM-4.7-Flash \
        --max-new-tokens 1000 \
        --step-limit 15
