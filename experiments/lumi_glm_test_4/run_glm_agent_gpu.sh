#!/bin/bash
#SBATCH --job-name=glm47_gpu
#SBATCH --partition=dev-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=03:00:00
#SBATCH --account=project_462001047
#SBATCH --output=glm47_gpu_%j.out

# Local GPU mode — loads GLM-4.7-Flash on LUMI GPU (~520s), then runs all tasks.
# myvenv has transformers 5.3.0.dev0 (supports glm4_moe_lite); prepended via PYTHONPATH.
module load LUMI/25.03
module use /appl/local/laifs/modules
module load lumi-aif-singularity-bindings

LAIF_SIF=/appl/local/laifs/containers/lumi-multitorch-latest.sif
MYVENV=/scratch/project_462001047/stockmj/myvenv
SIF_DIR=/scratch/project_462001047/stockmj/sif_images
SCRIPT_DIR=/scratch/project_462001047/stockmj/lumi_glm_test_4

export HF_HOME=/scratch/project_462001047/stockmj/hf_cache
export XDG_CACHE_HOME=$HF_HOME

# Extract SIFs to local /tmp (fast NVMe) before entering LAIF
SWE_TMP=/tmp/swe_glm_${SLURM_JOB_ID}
mkdir -p $SWE_TMP

for ID in astropy__astropy-12907 astropy__astropy-14365 django__django-10914; do
    SIF=$SIF_DIR/swe-bench.eval.x86_64.${ID}_latest.sif
    echo "=== Extracting $ID to /tmp ==="
    singularity build --sandbox $SWE_TMP/$ID $SIF
    echo "=== Done $ID ==="
done

srun singularity exec \
    --bind /scratch:/scratch \
    --bind /tmp:/tmp \
    --env PYTHONPATH="$MYVENV/lib/python3.12/site-packages" \
    --env HF_HOME="$HF_HOME" \
    --env XDG_CACHE_HOME="$HF_HOME" \
    "$LAIF_SIF" \
    python3 -u $SCRIPT_DIR/test_glm_agent.py \
        --tasks $SCRIPT_DIR/tasks_3.jsonl \
        --work-dir $SCRIPT_DIR/runs \
        --model zai-org/GLM-4.7-Flash \
        --max-new-tokens 1000 \
        --step-limit 20 \
        --swe-fs-dir "$SWE_TMP"
