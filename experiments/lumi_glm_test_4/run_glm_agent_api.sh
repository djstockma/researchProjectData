#!/bin/bash
#SBATCH --job-name=glm47_api
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --account=project_462001047
#SBATCH --output=glm47_api_%j.out

# API mode: no GPU needed, no model load — uses HF inference API (~2s/step).
# HF_TOKEN must be set in the environment before submitting:
#   HF_TOKEN=hf_... sbatch run_glm_agent_api.sh
module load LUMI/25.03
module use /appl/local/laifs/modules
module load lumi-aif-singularity-bindings

LAIF_SIF=/appl/local/laifs/containers/lumi-multitorch-latest.sif
SIF_DIR=/scratch/project_462001047/stockmj/sif_images
SCRIPT_DIR=/scratch/project_462001047/stockmj/lumi_glm_test_4

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
    --env HF_TOKEN="$HF_TOKEN" \
    "$LAIF_SIF" \
    python3 -u $SCRIPT_DIR/test_glm_agent.py \
        --tasks $SCRIPT_DIR/tasks_3.jsonl \
        --work-dir $SCRIPT_DIR/runs \
        --model zai-org/GLM-4.7-Flash \
        --max-new-tokens 1000 \
        --step-limit 20 \
        --use-api \
        --swe-fs-dir "$SWE_TMP"
