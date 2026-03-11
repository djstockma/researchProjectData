# Experiment 4: Mini-SWE-Agent on Real SWE-bench Tasks (LUMI)

## Goal

Run an interactive coding agent on real software engineering tasks from the
[SWE-bench Lite](https://github.com/SWE-bench/SWE-bench) benchmark, using the
LUMI supercomputer. Measure how many bugs the agent can fix (pass rate), and
compare performance between a locally-running GPU model and a remote API model.

---

## What is SWE-bench?

SWE-bench Lite is a benchmark of 300 real GitHub issues from popular Python
projects (astropy, Django, sympy, scikit-learn, etc.). Each task provides:

- A **problem statement** (the bug report / GitHub issue)
- A **repository** at a specific commit where the bug exists
- A set of **FAIL_TO_PASS tests** — tests that currently fail and must pass
  after the fix
- A set of **PASS_TO_PASS tests** — regression tests that must continue to pass

A solution is a git patch that fixes the bug.

---

## Agent Design: mini-SWE-agent

The agent follows a simple **observe → think → act** loop:

```
┌─────────────────────────────────────────────────────┐
│  System prompt: "You are a software engineer..."    │
│  Task prompt:   <bug description>                   │
└──────────────────────┬──────────────────────────────┘
                       │
              ┌────────▼────────┐
              │  LLM generates  │  ← GLM-4.7-Flash
              │  bash command   │
              └────────┬────────┘
                       │
              ┌────────▼────────┐
              │ Execute command │  ← in sandboxed repo
              │ in /testbed     │
              └────────┬────────┘
                       │
              ┌────────▼────────┐
              │ Feed output back│
              │ to LLM          │
              └────────┬────────┘
                       │
                  (repeat up to 20 steps)
                       │
              ┌────────▼────────┐
              │  git diff >     │
              │  patch.txt      │
              │  → submit       │
              └─────────────────┘
```

The LLM sees the full conversation history at each step. It must issue exactly
one bash command per response. The agent loop runs for up to 20 steps or until
the model submits a patch.

---

## Execution Environment

Each SWE-bench task has a Docker/Singularity container image with:
- The repository pre-checked out at the buggy commit (`/testbed`)
- The correct Python environment (conda env `testbed`, Python 3.9)
- No internet access

### LUMI setup

Running on LUMI requires two nested environments:

1. **LAIF container** (LUMI AI Factory, `lumi-multitorch-latest.sif`): Ubuntu 24,
   Python 3.12, provides PyTorch + the LLM
2. **SWE-bench container**: provides the buggy repository + test environment

Nested Singularity is not supported on LUMI (SUID stripped inside containers,
no FUSE, `unsquashfs` missing `liblzo2.so.2`).

### Solution: extract sandbox to local /tmp on compute node

Each SLURM job extracts SIF images to `/tmp` (local NVMe SSD on compute nodes)
**before** entering the LAIF container. This is fast (minutes vs. hours on Lustre)
and avoids nested container calls entirely.

```
SLURM job start (outside LAIF, on compute node):
  for each task:
    singularity build --sandbox /tmp/swe_fs/<id>  <id>.sif   # fast: local NVMe

  singularity exec --bind /tmp:/tmp  lumi-multitorch.sif \
      python3 test_glm_agent.py --swe-fs-dir /tmp/swe_fs
```

Inside the LAIF container, the agent runs bash commands directly against
the extracted directory using the sandboxed repo's Python environment:

```bash
unset PYTHONPATH   # prevent LAIF's Python 3.12 leaking into testbed's Python 3.9
PATH=/tmp/swe_fs/<id>/opt/miniconda3/envs/testbed/bin:$PATH \
    bash -c "cd /testbed && <agent_command>"
```

### Why not extract to Lustre?

Extracting a full SIF sandbox to Lustre (`/scratch`) creates millions of small
files (the full Ubuntu OS + conda env). Lustre has high latency for small-file
I/O: extraction takes 1+ hour per task. Local NVMe on compute nodes is orders
of magnitude faster (minutes per task).

---

## Model

**GLM-4.7-Flash** (`zai-org/GLM-4.7-Flash`) — a 9B parameter open-weight model
optimised for instruction following and code tasks.

Two inference modes were attempted:

| Mode | Where | Speed | Status |
|------|-------|-------|--------|
| **API** | HuggingFace Inference Router (free tier) | ~10s/step (rate-limited) | Ran, credits exhausted |
| **Local GPU** | LUMI `dev-g` partition, 1× AMD MI300X | — | Crashed (ROCm issue, see below) |

---

## Tasks

3 tasks used in this experiment:

| Instance ID | Repo | FAIL_TO_PASS | Notes |
|-------------|------|--------------|-------|
| astropy__astropy-12907 | astropy/astropy | 2 | Separability matrix in compound models |
| astropy__astropy-14365 | astropy/astropy | 1 | QDP format case sensitivity — unevaluable* |
| django__django-10914   | django/django   | 1 | FileResponse with pathlib.Path |

*astropy-14365: the FAIL_TO_PASS tests are added in the gold commit itself,
not at the base commit. The harness always reports FAIL even with the correct fix.
Confirmed correct fix (`re.IGNORECASE`) was produced in 4 out of 4 runs.*

---

## Files

| File | Purpose |
|------|---------|
| `test_glm_agent.py` | Agent harness (model loading, agent loop, sandbox execution) |
| `generate_tasks.py` | Downloads SWE-bench Lite and writes `tasks.jsonl` |
| `tasks.jsonl` | All 6 task definitions |
| `tasks_3.jsonl` | 3-task subset used in this experiment (on LUMI) |
| `run_glm_agent_api.sh` | SLURM job: API inference, `small` CPU partition |
| `run_glm_agent_gpu.sh` | SLURM job: local GPU inference, `dev-g` partition |
| `test_sandbox_exec.sh` | Infrastructure test (no model, validates Python env) |
| `runs/` | Per-run output: agent log, submitted patch, test output, metrics |

---

## How to Run

### Prerequisites (must be done each login on LUMI)

```bash
module load LUMI/25.03
module use /appl/local/laifs/modules
module load lumi-aif-singularity-bindings

LAIF_SIF=/appl/local/laifs/containers/lumi-multitorch-latest.sif
SIF_DIR=/scratch/project_462001047/stockmj/sif_images
```

### Pull SIF images (one-time, on login node)

```bash
for ID in astropy__astropy-12907 astropy__astropy-14365 django__django-10914; do
    singularity pull --dir $SIF_DIR \
        docker://ghcr.io/epoch-research/swe-bench.eval.x86_64.${ID}:latest
done
```

### Submit jobs

```bash
cd /scratch/project_462001047/stockmj/lumi_glm_test_4

# API run (no GPU needed, ~30-60 min total)
HF_TOKEN=hf_... sbatch run_glm_agent_api.sh

# GPU run (~2h total including model load)
sbatch run_glm_agent_gpu.sh
```

---

## Results

### API baseline (HF free tier, GLM-4.7-Flash, job 16665957)

| Instance ID | Result | Steps | Patch | Wall time | Notes |
|-------------|--------|-------|-------|-----------|-------|
| astropy__astropy-12907 | FAIL | 20 | ✗ | 214s | Hit step limit; explored code but couldn't locate fix |
| astropy__astropy-14365 | FAIL* | 17 | ✓ | 172s | Correct fix produced (re.IGNORECASE); unevaluable* |
| django__django-10914   | —    | —  | —  | —    | API credits exhausted mid-task |

API speed: ~10s/step (free tier rate-limited; expected ~2s/step).

### Local GPU (LUMI dev-g, 1× AMD MI300X, GLM-4.7-Flash, job 16665958)

| Instance ID | Result | Notes |
|-------------|--------|-------|
| all tasks | CRASHED | `RuntimeError: grouped gemm is not supported on ROCM` |

All 3 SIF extractions to `/tmp` succeeded (infrastructure works). Model loaded
in 1708s. Crashed on first inference call: GLM-4.7-Flash uses MoE grouped GEMM
(`torch._grouped_mm`) which is not implemented in the AMD ROCm PyTorch backend.

### Consistency check — astropy-14365 across all sessions

The agent was run on astropy-14365 four times across multiple sessions, always
producing the identical correct fix:

| Run | Steps | Mode | Wall time | Patch |
|-----|-------|------|-----------|-------|
| 20260311_064930 | 20 | GPU (local) | 639s | ✓ re.IGNORECASE |
| 20260311_071036 | 20 | GPU (local) | 506s | ✓ re.IGNORECASE |
| 20260311_150134 | 20 | API | 677s | ✓ re.IGNORECASE |
| 20260311_180123 | 17 | API | 172s | ✓ re.IGNORECASE |

The fix is found consistently — the FAIL result is purely the evaluation quirk.

---

## Open Issues / Next Steps

1. **ROCm grouped GEMM** — GLM-4.7-Flash (MoE architecture) crashes on AMD MI300X.
   Fix options:
   - Monkey-patch `torch._grouped_mm` to force loop fallback in moe.py
   - Use a non-MoE model (e.g. Llama-3.1-8B) that LUMI ROCm supports natively
   - Wait for upstream ROCm support for grouped GEMM

2. **API credits exhausted** — HF free tier ran out during django task.
   Options: purchase credits, use a different free provider, fix GPU first.

3. **Harder tasks need more steps** — astropy-12907 hit the 20-step limit.
   Increasing step limit or improving the prompt may help.

4. **Scale to full benchmark** — once GPU works, use SLURM job arrays to run
   all 300 SWE-bench Lite tasks in parallel (the original research goal).

*\* astropy-14365: FAIL_TO_PASS tests added in gold commit, not base commit.*
