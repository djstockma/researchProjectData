#!/bin/bash
# Quick test: verify that the pre-extracted SWE-bench sandbox can run Python
# commands from inside the LAIF container, WITHOUT a GPU or model load.
#
# Run on LUMI login node or as a short CPU batch job:
#   bash test_sandbox_exec.sh
#
# Or as a batch job (much faster allocation than dev-g):
#   sbatch --partition=small --nodes=1 --ntasks=1 --cpus-per-task=4 \
#          --mem=8G --time=00:10:00 --account=project_462001047 \
#          --output=test_sandbox_%j.out test_sandbox_exec.sh

module load LUMI/25.03
module use /appl/local/laifs/modules
module load lumi-aif-singularity-bindings

LAIF_SIF=/appl/local/laifs/containers/lumi-multitorch-latest.sif
SWE_FS=/scratch/project_462001047/stockmj/swe_fs/astropy__astropy-14365
TESTBED=$SWE_FS/testbed

echo "=== Testing sandbox execution (no model) ==="
echo "SWE_FS: $SWE_FS"
echo ""

if [ ! -d "$SWE_FS/testbed" ]; then
    echo "ERROR: Sandbox not found at $SWE_FS"
    echo "Run this first on the login node:"
    echo "  apptainer build --sandbox $SWE_FS /scratch/project_462001047/stockmj/sif_images/swe-bench.eval.x86_64.astropy__astropy-14365_latest.sif"
    exit 1
fi

singularity exec \
    --bind /scratch:/scratch \
    "$LAIF_SIF" \
    bash -c "
        unset PYTHONPATH
        export PATH=$SWE_FS/opt/miniconda3/envs/testbed/bin:$SWE_FS/opt/miniconda3/bin:\$PATH
        export LD_LIBRARY_PATH=$SWE_FS/opt/miniconda3/envs/testbed/lib:\$LD_LIBRARY_PATH
        cd $TESTBED

        echo '--- python version ---'
        python --version

        echo '--- astropy import ---'
        python -c 'import astropy; print(\"astropy\", astropy.__version__)'

        echo '--- git log (testbed repo) ---'
        git log --oneline -3

        echo '--- pytest dry run ---'
        python -m pytest astropy/io/fits/tests/test_connect.py -x -q --no-header --collect-only 2>&1 | head -20

        echo ''
        echo 'SUCCESS: sandbox execution works'
    "
