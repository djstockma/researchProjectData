#!/bin/bash
#SBATCH --job-name=glm47_swebench
#SBATCH --partition=dev-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --account=project_462001047
#SBATCH --output=glm47_swebench_%j.out

# LUMI AI Factory container — PyTorch 2.9.1+ROCm6.4, transformers 4.57.3 bundled
# myvenv at $MYVENV has transformers 5.3.0.dev0 (supports glm4_moe_lite); we
# prepend it via PYTHONPATH so it shadows the container's 4.57.3.
module load LUMI/25.03
module use /appl/local/laifs/modules
module load lumi-aif-singularity-bindings

LAIF_SIF=/appl/local/laifs/containers/lumi-multitorch-latest.sif
MYVENV=/scratch/project_462001047/stockmj/myvenv

# Paths
export HF_HOME=/scratch/project_462001047/stockmj/hf_cache
export XDG_CACHE_HOME=$HF_HOME
export SIF_DIR=/scratch/project_462001047/stockmj/sif_images
export SWE_FS_ROOT=/scratch/project_462001047/stockmj/swe_fs

# Run the agent inside the LAIF container.
# Nested Singularity does not work inside LAIF (no SUID, no FUSE, no user namespaces).
# Instead, SWE-bench SIFs are pre-extracted to $SWE_FS_ROOT/<instance_id>/ on the
# login node, and the agent runs commands directly using the extracted conda env.
# PYTHONPATH prepends myvenv's site-packages so transformers 5.3.0.dev0 shadows
# the container's 4.57.3 (which lacks glm4_moe_lite support).
srun singularity exec \
    --bind /scratch:/scratch \
    --env PYTHONPATH="$MYVENV/lib/python3.12/site-packages" \
    --env HF_HOME="$HF_HOME" \
    --env XDG_CACHE_HOME="$HF_HOME" \
    "$LAIF_SIF" \
    python3 -u /scratch/project_462001047/stockmj/lumi_glm_test_4/test_glm_agent.py \
        --tasks /scratch/project_462001047/stockmj/lumi_glm_test_4/tasks.jsonl \
        --work-dir /scratch/project_462001047/stockmj/lumi_glm_test_4/runs \
        --model zai-org/GLM-4.7-Flash \
        --max-new-tokens 2000 \
        --step-limit 30 \
        --swe-fs-dir "$SWE_FS_ROOT"

# -----------------------------------------------------------------------
# One-time setup on the LUMI login node (per SWE-bench instance):
#
#   export SIF_DIR=/scratch/project_462001047/stockmj/sif_images
#   export SWE_FS_ROOT=/scratch/project_462001047/stockmj/swe_fs
#
#   # Pull the SIF (if not already done):
#   cd $SIF_DIR
#   singularity pull docker://ghcr.io/epoch-research/swe-bench.eval.x86_64.astropy__astropy-14365:latest
#
#   # Extract SIF to sandbox directory (do this ONCE per instance):
#   mkdir -p $SWE_FS_ROOT
#   apptainer build --sandbox $SWE_FS_ROOT/astropy__astropy-14365 \
#       $SIF_DIR/swe-bench.eval.x86_64.astropy__astropy-14365_latest.sif
# -----------------------------------------------------------------------
