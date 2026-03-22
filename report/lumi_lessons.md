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

### `apptainer` is not available — use `singularity`

LUMI uses `singularity`, not `apptainer`. Commands using `apptainer exec` will simply fail
with "command not found". Always use `singularity exec`.

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

### `git diff` hangs on Lustre

Running `git diff` inside a repository on Lustre can hang indefinitely.
Use `grep` or `cat` to inspect file content instead. This affected early debugging — the
agent's submission step (which runs `git diff HEAD`) hit this issue in some configurations.

### First file copy to Lustre is slow — subsequent copies are fast

Copying the full QuixBugs repo (~200 MB) for the first task in a job took ~60–97s (cold
Lustre read). The same copy for subsequent tasks took 2–5s (warm OS page cache). This is
a Lustre cold-read anomaly, not representative of steady-state.

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
64 GB HBM — leaving no headroom for the KV cache. The job crashes with an OOM during the
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

**Fix:** One-time patch to `moe.py` in the myvenv — add a ROCm guard:

```python
elif hasattr(torch, "_grouped_mm") and not getattr(torch.version, "hip", None):
```

`torch.version.hip` is set on ROCm builds. This causes the code to use
`grouped_mm_fallback`, which iterates over experts using `torch.mm`. Functionally
correct, ~10–20% slower than the fused kernel would be.

### More GPUs do not speed up inference (MoE + ROCm fallback)

The `grouped_mm_fallback` path serialises expert routing — it loops over expert groups
regardless of GPU count. Splitting the model across 4 GPUs instead of 2 produces
**identical inference throughput**. The only measurable benefit of 4 GPUs is faster
model loading (2× speedup), because each chip holds a smaller weight shard and loads
faster from Lustre.

**Implication:** For long batch jobs where model load is amortised over many tasks,
2 GPUs is strictly more efficient per GPU-hour than 4.

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
on LUMI fails — this is a hard limitation of the current LUMI software stack.

**Consequence:** SWE-bench tasks cannot be run in GPU mode on LUMI without a
fundamental change to either the LUMI container setup or the SWE-bench evaluation method.

**Workaround used:** Switch to QuixBugs — a benchmark with no container requirements
that runs directly as plain Python in the LAIF environment.

---

## Agent Design

### Format error loops are expensive at ~100s/step

If the model generates an unparseable response (missing or multiple bash blocks), the
agent feeds back a format error message and retries. At ~100s/step, 14 consecutive
format errors waste ~1400s. A hard-reminder injection after 2 consecutive errors helps
most tasks, but some models / prompts are stuck in stable failure modes.

**Mitigation:** `--task-timeout 1800` — abandon the task after 30 minutes total.

### Pytest timeout is essential for infinite-loop bugs

Several QuixBugs programs have infinite recursion or infinite loops in the *buggy* state.
Running pytest on the unpatched file with no timeout hangs indefinitely, or until the
original 180s cap was hit (wasting 3 minutes per such task). Fix: `--timeout=30` on all
pytest calls.

### Per-task wall-time caps are essential for predictable throughput

Without a cap, a single bad task (format error loop + long context) can consume 35+ minutes
and prevent later tasks from running at all within the SLURM wall time. A 30-minute
per-task ceiling (`--task-timeout 1800`) sacrifices one task's solution quality in exchange
for predictable batch throughput — a good trade for measurement experiments.

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
`sbatch` jobs — each gets its own node, its own GPU allocation, and its own model load.
This is how the 2×2GPU parallel experiment was set up.

### Standard-g queue can have significant wait times

Jobs on `standard-g` (GPU partition) may wait several hours in the queue depending on
cluster load. Plan accordingly — submit jobs well before deadlines.

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
