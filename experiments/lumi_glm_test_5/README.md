# Experiment 5: Pipeline Timing and Resource Measurement

## Goal

Measure the time and resource consumption of **each phase** of the agent pipeline
separately, using a local GPU model on the LUMI supercomputer. Phases measured:

| Phase | Metric | What it covers |
|-------|--------|----------------|
| Model load | `model_load_time_s` | Loading weights from HF cache to GPU (session-level) |
| Task setup | `setup_time_s` | Copying the task repo + git init |
| LLM inference | `model_time_total_s` | Sum of all inference calls for the task |
| Command exec | `exec_time_total_s` | Sum of all bash command executions |
| Final test | `test_time_s` | Running pytest to verify the fix |
| Total | `total_wall_time_s` | Full wall time for the task |

Per-step breakdowns (`model_time_s`, `exec_time_s`) are in `step_log` inside each
task's `metrics.json`. Session-level data (including model load time) in `runs/session_meta.json`.

---

## Design Decisions and Pivots

### Why not SWE-bench?

The original plan was to run on SWE-bench Lite tasks. Two blockers:

**1. Nested Singularity on LUMI (fundamental)**
The local GPU model requires the LAIF container (for ROCm). SWE-bench tasks each
require their own Singularity container per agent step. Running `singularity exec`
from inside `singularity exec` is not supported on LUMI. There is no simple workaround
for GPU mode — see `DESIGN_NOTES.md` for the full analysis.

**2. Setup overhead**
Full sandbox extraction: ~1 h/task on Lustre. Testbed-only extraction (~2 min/task)
avoids this but still requires outside-LAIF execution, losing GPU access.

**Decision**: Use QuixBugs — no containers needed, runs directly in the LAIF environment.

### Why QuixBugs?

[QuixBugs](https://github.com/jkoppel/QuixBugs) is a published benchmark of 40
classic algorithm implementations each containing a single-line bug. It is:
- A real, citable benchmark (not custom toy tasks)
- Pure Python with pytest — zero container overhead
- Simple enough that the model reliably solves tasks, giving clean timing data
- Well-suited for measuring the agent loop rather than task difficulty

### ROCm grouped GEMM fix

GLM-4.7-Flash is a Mixture-of-Experts (MoE) model. With `transformers 5.3.0.dev0`,
it uses the `glm4_moe_lite` architecture which calls `torch._grouped_mm` for expert
routing. This function exists on ROCm PyTorch builds but is not implemented — it
crashes with:

```
RuntimeError: grouped gemm is not supported on ROCM
```

**Fix** (applied once to myvenv on LUMI): patch
`myvenv/lib/python3.12/site-packages/transformers/integrations/moe.py` to skip
`torch._grouped_mm` on ROCm and fall through to the existing loop-based fallback:

```python
# Before:
elif hasattr(torch, "_grouped_mm"):

# After:
elif hasattr(torch, "_grouped_mm") and not getattr(torch.version, "hip", None):
```

`torch.version.hip` is set on ROCm builds. This causes the code to skip the broken
fused kernel and use `torch.ops.transformers.grouped_mm_fallback` instead, which
iterates over expert groups using standard `torch.mm`. Inference is slower per step
than the fused kernel but functionally correct.

---

## Benchmark: QuixBugs (40 tasks)

Source: `task_repos/quixbugs/` (cloned from https://github.com/jkoppel/QuixBugs)

Each task is one buggy Python program with a single-line bug:
- Source files: `python_programs/<name>.py`
- Tests: `python_testcases/test_<name>.py` (pytest)
- Run from quixbugs root: `python3 -m pytest python_testcases/test_<name>.py -v`

All 40 programs: `bitcount`, `breadth_first_search`, `bucketsort`, `depth_first_search`,
`detect_cycle`, `find_first_in_sorted`, `find_in_sorted`, `flatten`, `gcd`,
`get_factors`, `hanoi`, `is_valid_parenthesization`, `kheapsort`, `knapsack`, `kth`,
`lcs_length`, `levenshtein`, `lis`, `longest_common_subsequence`, `max_sublist_sum`,
`mergesort`, `minimum_spanning_tree`, `next_palindrome`, `next_permutation`, `pascal`,
`possible_change`, `powerset`, `quicksort`, `reverse_linked_list`, `rpn_eval`,
`shortest_path_length`, `shortest_path_lengths`, `shortest_paths`, `shunting_yard`,
`sieve`, `sqrt`, `subsequences`, `to_base`, `topological_ordering`, `wrap`

---

## Agent Design

Same mini-SWE-agent loop as experiments 3 and 4:

```
Task description → LLM generates bash command → execute in task repo copy → feed output back → repeat
```

- 15 steps maximum per task
- Each task runs in a fresh copy of the quixbugs repo (`runs/<id>_<ts>/repo/`)
- Submit by echoing `COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT` + git diff patch

---

## Files

| File | Purpose |
|------|---------|
| `test_agent.py` | Agent harness with per-phase timing metrics |
| `tasks.jsonl` | 40 QuixBugs task definitions |
| `task_repos/quixbugs/` | Cloned QuixBugs repo |
| `run_agent_gpu.sh` | SLURM: `standard-g`, 1× AMD MI300X, 4 h, local GLM-4.7-Flash |
| `run_agent_api.sh` | SLURM: `small` CPU, outside LAIF, HF API + optional SWE-bench via singularity |
| `tasks_swe.jsonl` | SWE-bench django task definitions (for API mode if needed) |
| `generate_tasks.py` | Generates SWE-bench tasks.jsonl from HF dataset |
| `DESIGN_NOTES.md` | Detailed analysis of why SWE-bench + GPU doesn't work |
| `runs/` | Per-run output: `metrics.json`, `agent_log.jsonl`, `submitted_patch.diff`, `test_output.txt` |

---

## How to Run

### Each login: load modules

```bash
module load LUMI/25.03
module use /appl/local/laifs/modules
module load lumi-aif-singularity-bindings
```

### Sync files to LUMI

```bash
rsync -av --exclude 'runs/' -e "ssh -i ~/.ssh/id_ed25519_lumi" \
    experiments/lumi_glm_test_5/ \
    stockmj@lumi.csc.fi:/scratch/project_462001047/stockmj/lumi_glm_test_5/
```

### Submit GPU job

```bash
cd /scratch/project_462001047/stockmj/lumi_glm_test_5
sbatch run_agent_gpu.sh
squeue --me
```

### Monitor

```bash
tail -f glm5_gpu_<JOBID>.out
```

### Sync results back

```bash
rsync -av -e "ssh -i ~/.ssh/id_ed25519_lumi" \
    stockmj@lumi.csc.fi:/scratch/project_462001047/stockmj/lumi_glm_test_5/runs/ \
    experiments/lumi_glm_test_5/runs/
```

---

## Current Status (2026-03-20)

Two jobs running in parallel on LUMI `standard-g`, both over all 40 QuixBugs tasks:

| Job | ID | GPUs | Memory | Wall | Output dir |
|-----|----|------|--------|------|------------|
| 2 GPU | 16888661 | 2× MI250X (128 GB) | 160G | 8h | `runs/runs_2gpu/` |
| 4 GPU | 16888703 | 4× MI250X (256 GB) | 256G | 8h | `runs/runs_4gpu/` |

**Research question**: does splitting the model across 4 GPUs instead of 2 change inference speed?
With `device_map="auto"`, more GPUs = more inter-device communication per forward pass.
The ROCm fallback (sequential MoE routing) may amplify or reduce this effect.

Fixes applied vs job 16879200:
- pytest timeout 180s → 30s (was wasting 3 min on infinite-loop bugs)
- Format error hard reminder injected after 2 consecutive parse failures
- `repo/` directory deleted after each task (no more file bloat)
- 8h wall time (was 4h — only reached task 16/40 last time)

---

## Results

### GPU run — Job 16879200 (2026-03-20, `standard-g`, 2× AMD MI250X, 4h wall)

**Model load time: 1312s (22 min)**

16 of 40 tasks completed before 4h wall limit. **6 PASS / 10 FAIL (37.5% solve rate).**

| Task | Result | Steps | Wall (s) | Model (s) | Exec (s) | Setup (s) | Test (s) |
|------|--------|-------|----------|-----------|----------|-----------|----------|
| bitcount | **PASS** | 4 | 530 | 420 | 0.1 | 97 | 1.7 |
| breadth_first_search | **PASS** | 8 | 867 | 858 | 3.4 | 4.5 | 1.0 |
| bucketsort | **PASS** | 3 | 316 | 313 | 0.0 | 2.0 | 1.0 |
| depth_first_search | FAIL | 5 | 522 | 517 | 0.2 | 2.5 | 2.3 |
| detect_cycle | FAIL | 4 | 426 | 420 | 0.0 | 2.9 | 2.4 |
| find_first_in_sorted | FAIL | 2 | 389 | 207 | 0.0 | 2.1 | **180** ¹ |
| find_in_sorted | FAIL | 1 | **2174** | 2170 | 0.0 | 2.2 | 1.4 |
| flatten | **PASS** | 4 | 427 | 420 | 0.0 | 5.3 | 1.1 |
| gcd | **PASS** | 13 | 1812 | 1802 | 4.8 | 3.5 | 1.1 |
| get_factors | FAIL | 4 | 431 | 426 | 1.3 | 2.8 | 1.3 |
| hanoi | FAIL | 3 | 319 | 315 | 0.0 | 2.1 | 2.5 |
| is_valid_parenthesization | FAIL | 12 | 1559 | 1553 | 1.5 | 2.5 | 1.3 |
| kheapsort | **PASS** | 7 | 760 | 756 | 1.1 | 2.2 | 1.1 |
| knapsack | FAIL | 7 | 800 | 795 | 1.4 | 3.0 | 1.3 |
| kth | FAIL | 2 | 213 | 209 | 0.0 | 2.5 | 1.3 |
| lcs_length | FAIL | 4 | 417 | 413 | 0.0 | 2.5 | 1.4 |

¹ Test timeout: buggy code has infinite recursion → pytest hit 180s limit.
² `find_in_sorted` burned 2174s with only 1 logged step — model generated 14 unparseable responses (format error loop).

### Phase timing summary (16 tasks, job 16879200)

| Phase | Value | Notes |
|-------|-------|-------|
| **Model load** | **1313s (22 min)** | One-time per session; ROCm custom op init overhead |
| **Setup — 1st task** | **97s** | Cold Lustre copy of full quixbugs repo (~200 MB) |
| **Setup — subsequent** | **2.8s avg** | Warm cache, small delta |
| **Inference** | **115s/step avg** | ROCm fallback; grows with context (see below) |
| **Exec** | **1.5s avg total** | Local Python — essentially free |
| **Test** | **1.5s avg** | When code is correct; 180s cap on infinite loops |

#### Inference time grows with context length

Each agent step appends to the conversation, making the KV cache larger and inference slower:

| Steps in task | Avg inference/step |
|--------------|-------------------|
| 2 | ~103s |
| 3–4 | ~105s |
| 7–8 | ~107–114s |
| 12–13 | ~129–139s |

This is a direct consequence of the autoregressive KV cache growing with conversation length.

### Key observations

- **Inference dominates**: model time is 95–99% of task wall time. Exec and test are negligible.
- **Context growth increases inference cost**: a 13-step task pays ~35% more per step than a 2-step task.
- **First-task setup is a Lustre cold-copy anomaly** (97s vs 2–5s thereafter) — not representative of steady-state.
- **Format error loops are expensive**: `find_in_sorted` wasted 2174s with only 1 real step logged — the model generated 14 consecutive unparseable responses, each costing a full inference call (~2170s total wasted).
- **Test timeouts inflate results**: tasks where the buggy code has infinite recursion hit the 180s pytest cap (`find_first_in_sorted`). These should use a shorter timeout.
- **Model load slower than expected**: 22 min vs ~9 min in exp3, likely due to ROCm fallback custom op registration at import time and the much larger model (64 GB MoE vs ~12 GB dense in exp3).
- **Solve rate**: 6/16 PASS (37.5%) on the completed tasks.

### Remaining 24 tasks

Job hit the 4h wall limit during `levenshtein` (task 17). Still to run:
`levenshtein`, `lis`, `longest_common_subsequence`, `max_sublist_sum`, `mergesort`,
`minimum_spanning_tree`, `next_palindrome`, `next_permutation`, `pascal`,
`possible_change`, `powerset`, `quicksort`, `reverse_linked_list`, `rpn_eval`,
`shortest_path_length`, `shortest_path_lengths`, `shortest_paths`, `shunting_yard`,
`sieve`, `sqrt`, `subsequences`, `to_base`, `topological_ordering`, `wrap`

### Next steps

1. Submit follow-up job for remaining 24 tasks (8h wall time or two 4h runs).
2. Fix format error loops — add a hard format reminder after 2 consecutive parse failures.
3. Cap pytest timeout to 30s to avoid 180s waits on infinite-loop bugs.
4. Compile full 40-task timing table and produce phase-breakdown chart for final report.
