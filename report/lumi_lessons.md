# Lessons Learned: Running AI Agents on LUMI

A practical reference of everything that went wrong, was surprising, or had a non-obvious
solution during this project. Ordered roughly by discovery time.

---

## Environment & Modules

### Modules do not persist across sessions or compute nodes

Every new SSH login and every SLURM job starts with a clean module environment.
The required setup must be included in every SLURM script and re-run manually after
every login:

```bash
module load LUMI/25.03
module use /appl/local/laifs/modules
module load lumi-aif-singularity-bindings
```

Without these, `singularity` cannot bind the GPU-aware LAIF container correctly.

### Python packages must go in a persistent venv on scratch

The LAIF container provides a base Python environment but it is read-only and reset on
each job. Custom packages (e.g., `transformers 5.3.0.dev0`, `openai`) must be installed
into a virtual environment on `/scratch` and injected via `PYTHONPATH`:

```bash
--env PYTHONPATH="$MYVENV/lib/python3.12/site-packages"
```

Where `MYVENV=/scratch/project_462001047/stockmj/myvenv`.

---

## Lustre Filesystem

### HF model cache should live on scratch, not home

The HuggingFace cache (`HF_HOME`) must point to `/scratch`, not `$HOME`. Home quotas
are small (~20 GB), and the model weights alone are 64 GB.

```bash
export HF_HOME=/scratch/project_462001047/stockmj/hf_cache
export XDG_CACHE_HOME=$HF_HOME
```

---

## GPU & ROCm

### GLM-4.7-Flash OOMs on a single MI250X

GLM-4.7-Flash (MoE, bfloat16) needs ~64 GB for weights alone. A single MI250X chip has
64 GB HBM, leaving no headroom for the KV cache. The job crashes with an OOM during the
first inference call.

**Fix:** `device_map="auto"` + `--gpus-per-node=2` in SLURM. The model shards across
two chips (128 GB total), leaving ample room for KV cache.

### `torch._grouped_mm` is unimplemented on ROCm

`transformers 5.3.0.dev0` correctly recognises GLM-4.7-Flash as `glm4_moe_lite`
architecture and routes MoE expert dispatch through `torch._grouped_mm`. On AMD GPUs
with ROCm, this function exists in the namespace but raises:

```
RuntimeError: grouped gemm is not supported on ROCM
```

**Fix:** One-time patch to `moe.py` in the myvenv: add a ROCm guard:

```python
elif hasattr(torch, "_grouped_mm") and not getattr(torch.version, "hip", None):
```

`torch.version.hip` is set on ROCm builds. This causes the code to use
`grouped_mm_fallback`, which iterates over experts using `torch.mm`. Functionally
correct, ~10–20% slower than the fused kernel would be.


### Model load time varies significantly between jobs

Observed model load times on 2-GPU jobs: 22 min (job 16879200), 37.5 min (job 16888661).
The difference is likely Lustre cache state and node-to-node HBM bandwidth variance.
Model load time should be treated as a session-level overhead, not a fixed constant.

---

## Nested Singularity (SWE-bench blocker)

### Running `singularity exec` inside a Singularity container is not supported on LUMI

SWE-bench requires per-task Docker/Singularity containers for the Python environment.
The LAIF container (required for ROCm GPU access) already runs inside Singularity.
Attempting to call `singularity exec` from inside a running Singularity container
on LUMI fails. AFAIK, this is a hard limitation of the current LUMI software stack.

**Consequence:** SWE-bench tasks cannot be run in GPU mode on LUMI without a
fundamental change to either the LUMI container setup or the SWE-bench evaluation method.

Every workaround for the Experiment 5 timing runs was evaluated and failed:

| Approach | Problem |
|----------|---------|
| Full sandbox extraction to Lustre | ~1h/task setup -> prohibitively slow |
| Full sandbox extraction to `/tmp` (NVMe) | Up to 5h setup for 10 tasks; no wall time left for runs |
| Testbed-only extraction + `singularity exec` per step | Requires running *outside* LAIF → no ROCm → no GPU |
| API mode outside LAIF + SWE-bench | Works, but no local model load time to measure |

**Workaround used:** Switch to QuixBugs  (a benchmark with no container requirements
that runs directly as plain Python in the LAIF environment.)

---

## Agent Design

### Format error loops are expensive at ~100s/step

If the model generates an unparseable response (missing or multiple bash blocks), the
agent feeds back a format error message and retries. At ~100s/step, 14 consecutive
format errors waste ~1400s. A hard-reminder injection after 2 consecutive errors helps
most tasks, but some models / prompts are stuck in stable failure modes.

**Mitigation:** `--task-timeout 1800`  abandon the task after 30 minutes total.

### Pytest timeout is essential for infinite-loop bugs

Several QuixBugs programs have infinite recursion or infinite loops in the *buggy* state.
Running pytest on the unpatched file with no timeout hangs indefinitely, or until the
original 180s cap was hit (wasting 3 minutes per such task). Fix: `--timeout=30` on all
pytest calls.

### Per-task wall-time caps are essential for predictable throughput

Without a cap, a single bad task (format error loop + long context) can consume 35+ minutes
and prevent later tasks from running at all within the SLURM wall time. A 30-minute
per-task ceiling (`--task-timeout 1800`) sacrifices one task's solution quality in exchange
for predictable batch throughput..

### Context length grows inference cost ~50% across a task

Each agent step appends to the conversation, growing the KV cache. Step 14 costs ~50%
more per inference call than step 2. For timing experiments, tasks should be analysed
by step count, not just total wall time, to avoid confounding step count with model speed.

---

## Scheduling

### `nohup` + `&` keeps processes alive through SSH disconnects

Long-running processes launched in interactive sessions die when the SSH connection drops.
Use `nohup python3 script.py > out.log 2>&1 &` to detach from the session.

### Submit parallel jobs as separate `sbatch` calls, not job arrays

SLURM job arrays share a node allocation. For maximum parallelism, submit independent
`sbatch` jobs so each gets its own node, its own GPU allocation, and its own model load.
This is how the 2×2GPU parallel experiment was set up.


---

## Data Management

### Always exclude `repo/` when rsyncing results back

Each task run creates a `repo/` directory (full copy of the task repo). Syncing this
back wastes bandwidth and storage. Always use `--exclude 'repo/'` when rsyncing results:

```bash
rsync -av --exclude 'repo/' -e "ssh -i ~/.ssh/id_ed25519_lumi" \
    stockmj@lumi.csc.fi:/scratch/.../runs/ ./runs/
```

### The `repo/` directory is now deleted after each task

As of the current `test_agent.py`, `repo/` is removed with `shutil.rmtree` at the end
of each task. Only `metrics.json`, `agent_log.jsonl`, `submitted_patch.diff`, and
`test_output.txt` are retained.

### Do not SSH directly from development machine into LUMI for automated tasks

LUMI login nodes are not compute nodes and do not have GPUs. All heavy computation must
go through `sbatch`. Interactive debugging is possible on `dev-g` nodes via `salloc`,
but these are time-limited.
