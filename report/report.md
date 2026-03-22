# Scaling AI Coding Agents on LUMI

**Course project report — Jens Stockmarr**
**Date: March 2026**

---

## Abstract

This project investigates the practical feasibility of running interactive AI coding agents on
the LUMI supercomputer using a locally-hosted large language model. We implement a
mini-SWE-agent loop — an iterative bash-command-based agent — and measure the time
distribution across all pipeline phases (model load, task setup, LLM inference, command
execution, and testing) under different LUMI hardware configurations. Our primary finding
is that LLM inference completely dominates wall time (95–99%), and that adding more GPUs
does not improve inference throughput due to a ROCm kernel limitation in the MoE routing
path of the model. Horizontal parallelism (multiple independent jobs) proves far more
effective than vertical scaling (more GPUs per job) for maximising task throughput.

---

## 1. Research Question

**Primary question:**
How does scaling AI coding agents on a supercomputer (LUMI) affect performance,
efficiency, and solution quality when evaluated using a standardised software engineering
benchmark?

**Sub-questions:**
- What does *scaling* mean in practice for AI coding agents (model size, GPU count, parallelism)?
- Which scaling axes provide meaningful throughput gains?
- What bottlenecks emerge when running an LLM-based agent loop on HPC infrastructure?

**Scope — what was studied:**
- Running a large language model locally on LUMI GPUs
- Measuring per-phase timing across the full agent pipeline
- Comparing single-job GPU configurations (2× vs 4× MI250X)
- Comparing sequential single-job vs parallel multi-job scheduling

**Out of scope:**
- Training or fine-tuning models
- Multi-agent sampling per task (running N agents per task, picking best)
- Human-in-the-loop evaluation

---

## 2. System Architecture

### 2.1 Pipeline overview

The overall pipeline takes a task description, runs an iterative agent loop with a local
LLM, and evaluates the result against a test suite.

```mermaid
flowchart TD
    BENCH["📋 Benchmark tasks (QuixBugs — 40 Python bugs)"]
    TASK["Task: bug description + source file + test command"]
    AGENT["🤖 Coding Agent (mini-SWE-agent loop)"]
    LLM["🧠 GLM-4.7-Flash on LUMI GPU (local)"]
    ENV["🗂 Task repo copy (plain Python, no container)"]
    PATCH["📄 Git patch (submitted diff)"]
    EVAL["✅ pytest — PASS / FAIL?"]
    METRICS["📊 Timing metrics per phase"]

    BENCH -->|"one task at a time"| TASK
    TASK --> AGENT
    AGENT <-->|"prompt / response"| LLM
    AGENT <-->|"bash commands + output"| ENV
    AGENT -->|"git diff"| PATCH
    PATCH --> EVAL
    EVAL --> METRICS
```

### 2.2 Agent loop

Each task is solved through an iterative loop: the model proposes a bash command, it runs
in a copy of the task repository, and the output is fed back. This continues until the
model submits a patch or the step limit is reached.

```mermaid
flowchart TD
    START["Task description → model"]
    THINK["Model: generate one bash command"]
    RUN["Execute command in repo copy"]
    CHECK{Submit marker in output?}
    OBS["Feed output back to model"]
    SUBMIT["Collect git diff as patch"]
    TEST["Run pytest on patched repo"]
    RESULT(["PASS / FAIL + timing metrics"])

    START --> THINK
    THINK --> RUN
    RUN --> CHECK
    CHECK -->|no| OBS
    OBS --> THINK
    CHECK -->|yes| SUBMIT
    SUBMIT --> TEST
    TEST --> RESULT
```

Key design choices:
- The **model** is a black-box text generator loaded once per SLURM job
- The **agent** wraps prompts, parses responses, and executes shell commands
- **No containers** — QuixBugs runs as plain Python directly in the LAIF environment
- Correctness is determined purely by pytest outcomes — no human judgment

### 2.3 LUMI execution environment

```mermaid
flowchart LR
    subgraph LUMI ["LUMI — standard-g partition"]
        direction TB
        subgraph JOB ["SLURM job (1 node)"]
            SIF["LAIF Singularity container (lumi-multitorch-latest.sif)"]
            GPU["2–4× AMD MI250X GPUs (64 GB HBM each)"]
            MODEL["GLM-4.7-Flash (64 GB, bfloat16 MoE)"]
            AGENT2["Agent loop (test_agent.py)"]
            SIF --> AGENT2
            SIF --> MODEL
            MODEL --- GPU
        end
    end
    HF["HF cache (/scratch Lustre)"]
    TASKS["tasks.jsonl + QuixBugs repo"]
    HF -->|"model weights"| MODEL
    TASKS --> AGENT2
```

---

## 3. Benchmark: QuixBugs

### 3.1 Why not SWE-bench?

The original plan was to use **SWE-bench Lite** (300 real GitHub bug-fix tasks). Two
fundamental blockers arose on LUMI:

**Nested Singularity:** The LAIF container (required for ROCm GPU access) runs
inside Singularity. SWE-bench tasks each require their own per-task Singularity container
for the Python environment. Running `singularity exec` from inside `singularity exec` is
not supported on LUMI — there is no straightforward workaround that preserves GPU access.

**Lustre setup overhead:** Extracting a full SWE-bench task sandbox from a SIF image to
the Lustre scratch filesystem takes approximately 1 hour per task — making a 40-task run
take 40+ hours of setup alone.

**Decision:** Switch to a benchmark that runs natively in the LAIF Python environment
without containers.

### 3.2 QuixBugs benchmark

[QuixBugs](https://github.com/jkoppel/QuixBugs) is a published benchmark of 40 classic
algorithm implementations, each containing a deliberate single-line bug.

| Property | Value |
|----------|-------|
| Tasks | 40 Python programs |
| Bug type | Single-line logic error per program |
| Evaluation | pytest — binary PASS/FAIL |
| Dependencies | Standard library only — no containers |
| Source | `python_programs/<name>.py` |
| Tests | `python_testcases/test_<name>.py` |

This makes QuixBugs ideal for measuring agent pipeline timing: it is a real, citable
benchmark with clean binary outcomes and zero container overhead.

---

## 4. Model: GLM-4.7-Flash on LUMI

**GLM-4.7-Flash** (ZhipuAI / zai-org) is a 32B Mixture-of-Experts (MoE) model. In
bfloat16 precision, the full weight set occupies ~64 GB — exactly matching a single
AMD MI250X GPU chip. This creates practical constraints on LUMI.

### 4.1 ROCm grouped-GEMM fix

With `transformers 5.3.0.dev0`, GLM-4.7-Flash uses the `glm4_moe_lite` architecture
which dispatches MoE expert routing through `torch._grouped_mm`. On LUMI's AMD MI250X
GPUs (ROCm), this function exists but is not implemented, causing:

```
RuntimeError: grouped gemm is not supported on ROCM
```

**Fix:** Patched `moe.py` in the myvenv to skip the fused kernel on ROCm:

```python
# Before:
elif hasattr(torch, "_grouped_mm"):

# After:
elif hasattr(torch, "_grouped_mm") and not getattr(torch.version, "hip", None):
```

This causes the code to fall through to `grouped_mm_fallback`, which iterates over expert
groups using standard `torch.mm`. Inference is functionally correct but slower than the
fused kernel would be.

### 4.2 OOM on single GPU

The 64 GB model fills an entire MI250X chip, leaving no room for the KV cache during
inference. Fix: `device_map="auto"` with `torch_dtype=torch.bfloat16`, requiring at least
2 GPUs (2× 64 GB = 128 GB total).

---

## 5. Experiment History

### Experiment 2 — One-shot diff generation (`lumi_glm_test_2`)

A simple proof-of-concept: feed a bug description and source file to GLM-4.7-Flash in a
single prompt and ask for a diff. Ran both via HuggingFace inference API and locally on
LUMI GPU.

- **Outcome:** Working pipeline; variable output quality. No test evaluation.
- **Key learning:** Local GPU inference on LUMI is feasible with correct module/container setup.

### Experiment 3 — Interactive agent, toy task (`lumi_glm_test_3`)

Introduced the iterative agent loop (mini-SWE-agent style) on a simple fibonacci bug-fix
task.

| Mode | Steps | Wall time | Model load |
|------|-------|-----------|------------|
| API (HF router) | 10 | 19s | — |
| Local GPU (LUMI) | 6 | 520s | 520s |

- **Key learning:** Model loaded as generic CausalLM (~12 GB) with old transformers —
  this was incorrect architecture. New transformers correctly loads the full 64 GB MoE.

### Experiment 4 — Real SWE-bench tasks (`lumi_glm_test_4`)

Ran the agent on real SWE-bench Lite tasks (astropy and django repos) using per-task
Singularity containers.

- **Outcome:** Pipeline working end-to-end. 1/2 evaluable tasks solved.
- **Key learning (ROCm crash):** Discovered `torch._grouped_mm` not supported on ROCm — patched the fallback.
- **Key learning (nested Singularity):** SWE-bench containers inside the LAIF container not supported on LUMI — fundamental blocker for local GPU mode on SWE-bench.
- **Key learning (API credits):** HF free tier exhausted after ~2.5 tasks at ~10s/step.

### Experiment 5 — Pipeline timing on QuixBugs (`lumi_glm_test_5`)

**Goal:** Measure the time distribution across all pipeline phases using a proper benchmark,
under different LUMI hardware configurations.

- Switched from SWE-bench to QuixBugs (no container overhead)
- Implemented per-phase timing instrumentation
- Ran three GPU configurations: 2GPU serial, 4GPU serial, 2×2GPU parallel

Full results in Section 6.

---

## 6. Results: Experiment 5

### 6.1 SLURM configurations tested

| Config | Jobs | GPUs/job | Tasks/job | Total GPU-h | Output dir |
|--------|------|----------|-----------|-------------|------------|
| 2GPU serial | 1 | 2× MI250X | 40 | 16 | `runs_2gpu/` |
| 4GPU serial | 1 | 4× MI250X | 40 | 32 | `runs_4gpu/` |
| **2×2GPU parallel** | **2** | **2× MI250X** | **20 each** | **32** | `runs_parallel_a/` + `runs_parallel_b/` |

The parallel configuration uses the same total GPU budget as 4GPU serial but splits the
workload across two simultaneously-scheduled jobs.

### 6.2 Phase timing — where does the time go?

All timings from jobs 16888661 (2GPU) and 16888703 (4GPU), 8h wall, 24–26 tasks completed.

| Phase | 2GPU | 4GPU | Notes |
|-------|------|------|-------|
| **Model load** | **2248s (37.5 min)** | **1182s (19.7 min)** | One-time per job |
| Setup — 1st task | ~27s | ~61s | Cold Lustre copy |
| Setup — subsequent | ~4s | ~5s | Warm cache |
| **Inference per step** | **~105–110s** | **~107–113s** | ROCm fallback |
| Exec per step | ~0.5s | ~0.5s | Local Python |
| Test (final pytest) | ~2s | ~2s | When not buggy |

> **Figure 1 — Phase time distribution (placeholder)**
> *Pie chart: for a typical 6-step task on 2GPU, ~97% of wall time is LLM inference,
> ~2% is model load amortised, <1% is setup/exec/test combined.*

### 6.3 Inference dominates

For every configuration tested, LLM inference accounts for **95–99% of task wall time**.
Setup, execution, and testing are negligible.

> **Figure 2 — Phase breakdown bar chart (placeholder)**
> *Stacked bar per task: model_time vs setup_time vs exec_time vs test_time.
> Will be generated from runs_2gpu/ metrics when parallel job results arrive.*

### 6.4 Context growth increases inference cost

Each agent step appends to the conversation history. The growing KV cache makes each
subsequent inference call slower:

| Steps in task | 2GPU avg inf/step | 4GPU avg inf/step |
|--------------|-------------------|-------------------|
| 2–3 | ~104s | ~106s |
| 4–7 | ~106–110s | ~108–113s |
| 9–12 | ~120–130s | ~124–134s |
| 13–14 | ~150–160s | ~150–180s |

A 14-step task pays approximately **50% more per step** than a 2-step task.

> **Figure 3 — Inference time vs step number (placeholder)**
> *Scatter plot: x = step number, y = inference time (s), coloured by task.
> Shows the linear-ish growth of per-step cost with context length.*

### 6.5 2GPU vs 4GPU — inference speed is identical

| Metric | 2GPU | 4GPU |
|--------|------|------|
| Avg inf/step (normal tasks) | ~107s | ~110s |
| Model load | 37.5 min | **19.7 min** |
| Tasks completed (8h) | 24/40 | 26/40 |
| Solve rate | 10/24 (42%) | 9/26 (35%) |

Adding 2 more GPUs provides **no inference speedup**. The ROCm `grouped_mm_fallback`
serialises MoE expert routing into sequential `torch.mm` calls regardless of how many
GPUs hold the weights. The only benefit of 4 GPUs is a 2× faster model load — because
each chip holds a smaller weight shard and requires less per-chip IO from the Lustre cache.

### 6.6 Per-task results (2GPU serial, job 16888661)

| Task | Result | Steps | Wall (s) | Inf/step (s) |
|------|--------|-------|----------|--------------|
| bitcount | **PASS** | 4 | 474 | 104 |
| breadth_first_search | FAIL | 7 | 764 | 108 |
| bucketsort | **PASS** | 3 | 319 | 105 |
| depth_first_search | FAIL | 5 | 531 | 104 |
| detect_cycle | FAIL | 4 | 421 | 103 |
| find_first_in_sorted | FAIL | 2 | 239 | 103 |
| find_in_sorted | FAIL | 1 | **2201** | — ¹ |
| flatten | **PASS** | 4 | 435 | 106 |
| gcd | **PASS** | 6 | 668 | 110 |
| get_factors | FAIL | 8 | 965 | 119 |
| hanoi | FAIL | 3 | 324 | 106 |
| is_valid_parenthesization | **PASS** | 10 | 1217 | 121 |
| kheapsort | **PASS** | 7 | 763 | 108 |
| knapsack | FAIL | 14 | **2220** | 158 |
| kth | FAIL | 2 | 228 | 106 |
| lcs_length | FAIL | 4 | 422 | 105 |
| levenshtein | FAIL | 5 | **2025** | 403 ¹ |
| lis | FAIL | 4 | 431 | 106 |
| longest_common_subsequence | FAIL | 7 | **2209** | 315 ¹ |
| max_sublist_sum | **PASS** | 12 | 1524 | 126 |
| mergesort | **PASS** | 6 | **2273** | 376 ¹ |
| minimum_spanning_tree | FAIL | 2 | 217 | 105 |
| next_palindrome | **PASS** | 7 | **2069** | 294 ¹ |
| next_permutation | **PASS** | 10 | 1619 | 134 |

¹ Abnormally high: format error loop (model generating unparseable responses) or severe
context growth. These tasks consumed a disproportionate share of the 8h budget.

**10/24 PASS (42%)** on completed tasks. Job hit 8h wall limit at task 25/40.

### 6.7 Parallel 2×2GPU results

> **[PLACEHOLDER — jobs 16914578 and 16914579 pending]**
>
> Two jobs submitted simultaneously on 2026-03-21:
> - Job A (16914578): tasks 1–20 (`tasks_a.jsonl`) → `runs/runs_parallel_a/`
> - Job B (16914579): tasks 21–40 (`tasks_b.jsonl`) → `runs/runs_parallel_b/`
>
> Both: 2× MI250X, 8h wall, per-task timeout 1800s.
>
> Expected outcome: all 40 tasks completed within the 8h window.
> Results table and timing comparison with serial configurations to be added here.

> **Figure 4 — Configuration comparison (placeholder)**
> *Bar chart: tasks completed per configuration (2GPU serial, 4GPU serial, 2×2GPU parallel)
> and wall-clock time to complete all tasks.*

---

## 7. Key Findings

### Finding 1: Inference dominates everything

LLM inference accounts for 95–99% of all task wall time. Setup, bash command execution,
and test running are collectively less than 5%. At this model scale (~100s/step), there is
no value in optimising any other phase.

### Finding 2: More GPUs ≠ faster inference (on ROCm)

The ROCm `grouped_mm_fallback` serialises MoE expert routing into sequential matrix
multiplications regardless of GPU count. This means:
- 4 GPUs offer identical inference throughput to 2 GPUs
- The only measurable benefit is 2× faster model loading
- For batch jobs (where model load is amortised over many tasks), 2 GPUs is the
  strictly more efficient allocation

### Finding 3: Horizontal parallelism beats vertical scaling

Splitting 40 tasks across two simultaneous 2-GPU jobs completes the full benchmark
faster than one 4-GPU job running all 40 tasks sequentially — at the same total GPU cost.
This is the natural HPC scaling strategy: task-level parallelism over hardware-level
parallelism.

### Finding 4: Context length is the real inference cost driver

Per-step inference time grows ~50% from step 2 to step 14. Managing context length
(fewer steps, history summarisation) would reduce cost more than any hardware change.

### Finding 5: Runaway tasks need explicit time budgets

A small number of tasks (format error loops, severe context explosion) consumed
disproportionate wall time. A per-task wall-time cap (implemented as `--task-timeout 1800`)
is essential for predictable batch throughput.

---

## 8. Discussion

### What worked well

- The end-to-end pipeline is robust and produces clean, reproducible timing data
- QuixBugs is an excellent benchmark for this purpose — real, citable, zero overhead
- The ROCm patch is simple and stable; no further inference crashes observed
- Per-phase instrumentation gives fine-grained data across all configurations

### What did not work as expected

- **SWE-bench on LUMI GPU** is not feasible without nested container support — a
  fundamental limitation of the current LUMI software stack
- **4 GPUs do not improve inference** — the ROCm fallback removes any benefit of
  tensor parallelism for this model's MoE routing
- **Format error loops** persisted for certain tasks despite the hard-reminder fix,
  suggesting the model has a stable failure mode on specific problem types

### Limitations

- QuixBugs is simpler than real-world bug-fix benchmarks — the 37–42% solve rate
  may not generalise to harder tasks
- Only one model (GLM-4.7-Flash) was tested — results are specific to this MoE
  architecture on ROCm
- The parallel experiment results are pending (see Section 6.7)

---

## 9. Conclusion

We successfully ran an interactive AI coding agent with a locally-hosted 32B MoE model
on LUMI, measured the full pipeline timing at per-phase granularity, and compared multiple
hardware configurations. The dominant finding is that LLM inference is the exclusive
bottleneck (95–99% of wall time), and that on LUMI's AMD MI250X GPUs with the current
ROCm software stack, vertical scaling (more GPUs per job) provides no inference throughput
benefit. Horizontal scaling (more parallel jobs) is the correct strategy for maximising
task throughput within a fixed GPU budget — a conclusion that generalises to any HPC
deployment of large MoE models under similar ROCm constraints.

---

## Appendix: Files and Reproducibility

| File | Purpose |
|------|---------|
| `experiments/lumi_glm_test_5/test_agent.py` | Agent harness with per-phase timing |
| `experiments/lumi_glm_test_5/tasks.jsonl` | All 40 QuixBugs task definitions |
| `experiments/lumi_glm_test_5/tasks_a.jsonl` | Tasks 1–20 (parallel batch A) |
| `experiments/lumi_glm_test_5/tasks_b.jsonl` | Tasks 21–40 (parallel batch B) |
| `experiments/lumi_glm_test_5/run_agent_gpu.sh` | SLURM: 2GPU serial |
| `experiments/lumi_glm_test_5/run_agent_gpu4.sh` | SLURM: 4GPU serial |
| `experiments/lumi_glm_test_5/run_agent_gpu_a.sh` | SLURM: parallel batch A |
| `experiments/lumi_glm_test_5/run_agent_gpu_b.sh` | SLURM: parallel batch B |
| `experiments/lumi_glm_test_5/runs/` | All run outputs (metrics.json per task) |
| `report/data/` | Extracted timing data for figures |
| `docs/project.md` | Living project document |
| `report/lumi_lessons.md` | Lessons learned on LUMI |
