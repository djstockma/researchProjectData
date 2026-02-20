# Notes from first meeting with TA and also session 30.1

## Starting plan (made with TA)
1. Supercomputer basics
   * How to use LUMI
   * Allocate resource
   * Queue jobs

2. Frameworks investigation & code hands on.
* Start with simple coding agent:
  * https://mini-swe-agent.com/latest/
  * https://github.com/SWE-bench/SWE-bench
  * https://www.tbench.ai/

3. Introduce basic profiling tools. Think up how to analyze results.
Overview - mini-SWE-agent documentation



## Progress first week
* LUMI access
* Running basic script in LUMI

## Learnt from session 30.1
* Possibility for opposition or teamwork with other project on same topic
* Importance of definging the term "scaling". Before I move on to the actual benchmarking, I need to consider what I here mean as scaling. Is it scaling n of nodes only? One agent or many? THis hsould be thought about, decided, visualized so it is clear

## Next goals
1. Look at docs of some simple model
2. Figure out what "scaling" means for said model. Visualize and define this (might change later as I move on)
3. Try simple benchmarks / measuring efficiency, resource use etc. Define also here what we actually want to look at / study

*** To figure out: how to take docker SWE benchmark and run in apptainer / on lumi


# Meeting 6.2

## Progress:
* Ran mini-SWE and SWE-Bench locally 


## Next: 
* How to scale
* Defining scope
* How to run
* Using models

* Running a model on LUMI: lumi has preconfigured pytorch container
  * Starting with [GLM 4.7 flash](https://huggingface.co/zai-org/GLM-4.7-Flash)

* Ask about collaboration

* Do it as a jupyter notebook job

* Goal: reduce end-2-end-latency for larger benchmark


## 12.2
have to use ssh instead of web cli. Goal: get model running locally!

Status:
### LUMI PyTorch GPU Test Summary

* Loaded the correct LUMI software stack and CSC PyTorch Singularity module:

  ```bash
  module load LUMI/25.03
  module use /appl/local/csc/modulefiles/
  module load pytorch
  ```
* Created a simple PyTorch test script (`test_gpu.py`) that checks GPU availability and performs a small computation on the GPU.
* Submitted the script as a Slurm job from the login node using:

  ```bash
  salloc -N1 -p standard-g -t 00:10:00
  srun -N1 -n1 --gpus 1 python3 test_gpu.py
  ```
* Verified the job ran successfully on a GPU node by checking the output file:

  ```text
  PyTorch version: 2.7.1+rocm6.2.4
  CUDA available: True
  Number of GPUs detected: 1
  Random tensor on GPU:
  tensor([...], device='cuda:0')
  ```
* Confirmed that PyTorch can access the GPU and execute computations inside the CSC Singularity container.

Next step: adapt this workflow to run an actual training model on LUMI.


## Workflow for lumi

eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_ed25519_lumi

rsync -av --exclude 'runs/' \
  -e "ssh -i ~/.ssh/id_ed25519_lumi" \
  /home/jens/researchProjectData/experiments/lumi_glm_test_2/ \
  stockmj@lumi.csc.fi:/scratch/project_462001047/stockmj/lumi_glm_test_2/

