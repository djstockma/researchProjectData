#!/bin/bash
#SBATCH --job-name=glm47
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --account=project_462001047
#SBATCH --output=glm47_%j.out

# Load LUMI PyTorch environment
module load LUMI/25.03
module use /appl/local/csc/modulefiles/
module load pytorch

# Setup scratch paths
export HF_HOME=/scratch/project_462001047/stockmj/hf_cache
export XDG_CACHE_HOME=$HF_HOME
export PYTHONUSERBASE=/scratch/project_462001047/stockmj/python_user
export PATH=$PYTHONUSERBASE/bin:$PATH

# Ensure pip is up-to-date and install required packages
python3 -m pip install --upgrade pip --user
python3 -m pip install --user git+https://github.com/huggingface/transformers.git
python3 -m pip install --user accelerate

# Run the Python test
srun python3 test_glm.py