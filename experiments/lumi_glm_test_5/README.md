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

## Results

### Run summary

Three GPU jobs completed on LUMI `standard-g` using GLM-4.7-Flash (local, ROCm).
Job 16879200 was the initial 4h test run; jobs 16888661 and 16888703 are the main
2GPU vs 4GPU comparison runs (8h wall each).

| Job | GPUs | Wall | Tasks done | PASS | FAIL | Model load | Output dir |
|-----|------|------|------------|------|------|------------|------------|
| 16879200 | 2× MI250X | 4h | 16/40 | 6 (38%) | 10 | 22 min | `runs_2gpu_job16879200/` |
| 16888661 | 2× MI250X | 8h | 24/40 | 10 (42%) | 14 | **37.5 min** | `runs_2gpu/` |
| 16888703 | 4× MI250X | 8h | 26/40 | 9 (35%) | 17 | **19.7 min** | `runs_4gpu/` |

Neither 8h job completed all 40 tasks — a handful of runaway tasks (format error loops,
long context) consumed a disproportionate share of the wall time budget.

---

### Job 16879200 — Initial run (2GPU, 4h wall)

**Model load: 1312s (22 min).** 16/40 tasks before wall limit. 6 PASS / 10 FAIL (38%).

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

¹ Test timeout: infinite recursion in buggy code → pytest hit 180s limit (later fixed to 30s).
² `find_in_sorted` burned 2174s on 1 logged step — 14 consecutive unparseable responses, each
  costing a full inference call. Format error hard-reminder fix added for subsequent jobs.

---

### Jobs 16888661 vs 16888703 — 2GPU vs 4GPU comparison (8h wall)

Fixes vs job 16879200: pytest timeout 30s, format hard-reminder after 2 parse failures,
`repo/` deleted after each task, 8h wall time.

**Model load: 2GPU = 2248s (37.5 min), 4GPU = 1182s (19.7 min)**

| Task | 2GPU result | 2GPU wall | 2GPU inf/step | 4GPU result | 4GPU wall | 4GPU inf/step |
|------|-------------|-----------|---------------|-------------|-----------|---------------|
| bitcount | **PASS** | 474s | 104s | **PASS** | 500s | 107s |
| breadth_first_search | FAIL | 764s | 108s | **PASS** | 799s | 113s |
| bucketsort | **PASS** | 319s | 105s | **PASS** | 327s | 107s |
| depth_first_search | FAIL | 531s | 104s | FAIL | 535s | 106s |
| detect_cycle | FAIL | 421s | 103s | FAIL | 428s | 105s |
| find_first_in_sorted | FAIL | 239s | 103s ² | FAIL | 244s | 106s ² |
| find_in_sorted | FAIL | **2201s** | — ³ | FAIL | **2308s** | — ³ |
| flatten | **PASS** | 435s | 106s | **PASS** | 439s | 108s |
| gcd | **PASS** | 668s | 110s | FAIL | 435s | 107s |
| get_factors | FAIL | 965s | 119s | FAIL | 1967s | 151s |
| hanoi | FAIL | 324s | 106s | FAIL | 359s | 109s |
| is_valid_parenthesization | **PASS** | 1217s | 121s | **PASS** | 1128s | 124s |
| kheapsort | **PASS** | 763s | 108s | **PASS** | 800s | 113s |
| knapsack | FAIL | **2220s** | 158s | **PASS** | 1214s | 134s |
| kth | FAIL | 228s | 106s | FAIL | 220s | 107s |
| lcs_length | FAIL | 422s | 105s | FAIL | 429s | 106s |
| levenshtein | FAIL | **2025s** | 403s ³ | FAIL | **2192s** | 435s ³ |
| lis | FAIL | 431s | 106s | FAIL | 450s | 109s |
| longest_common_subsequence | FAIL | **2209s** | 315s ³ | FAIL | **2332s** | 179s |
| max_sublist_sum | **PASS** | 1524s | 126s | FAIL | **2353s** | 578s ³ |
| mergesort | **PASS** | **2273s** | 376s ³ | **PASS** | 962s | 118s |
| minimum_spanning_tree | FAIL | 217s | 105s | FAIL | 217s | 107s |
| next_palindrome | **PASS** | **2069s** | 294s ³ | **PASS** | 919s | 114s |
| next_permutation | **PASS** | 1619s | 134s | FAIL | 1004s | 125s |
| pascal | — | — | — | FAIL | **2211s** | — ³ |
| possible_change | — | — | — | FAIL | **2334s** | — ³ |

² `find_first_in_sorted`: 30s pytest timeout hit (buggy infinite recursion).
³ Abnormally high inference/step — context explosion or format error loop burning tokens.

---

### Phase timing analysis

#### 2GPU vs 4GPU comparison

| Phase | 2GPU (job 16888661) | 4GPU (job 16888703) | Notes |
|-------|---------------------|---------------------|-------|
| **Model load** | **2248s (37.5 min)** | **1182s (19.7 min)** | 2× faster with 4 GPUs |
| Setup — 1st task | 27s | 61s | Cold Lustre copy |
| Setup — subsequent | ~4s avg | ~5s avg | Warm cache |
| **Inference (normal tasks)** | **~105–110s/step** | **~106–113s/step** | Essentially identical |
| Exec | ~3s avg total | ~3s avg total | Negligible |
| Test | ~2s avg | ~2s avg | Negligible (30s cap) |

#### Inference grows with context

Inference time per step scales with KV cache size (accumulated conversation history):

| Steps taken | 2GPU avg inf/step | 4GPU avg inf/step |
|-------------|-------------------|-------------------|
| 2–3 | ~104s | ~106s |
| 4–7 | ~106–110s | ~108–113s |
| 9–12 | ~120–130s | ~124–134s |
| 13–14 | ~150–160s | ~150–180s |

A 14-step task pays roughly **50% more per step** than a 2-step task — direct consequence of
autoregressive KV cache growth.

#### Phase breakdown (% of wall time, normal tasks)

| Phase | Share |
|-------|-------|
| LLM inference | 95–99% |
| Setup | <1% (subsequent tasks) |
| Exec | <1% |
| Test | <1% |

**Inference completely dominates.** All other phases are negligible at this model scale.

---

### Key findings

1. **Inference speed is identical across 2 and 4 GPUs (~107s/step for normal tasks).**
   The ROCm grouped-GEMM fallback serialises MoE expert routing into sequential `torch.mm`
   calls regardless of how many GPUs are available. Adding more GPUs does not parallelise
   the bottleneck — it only splits the weight storage.

2. **Model load is 2× faster on 4 GPUs** (19.7 min vs 37.5 min). More VRAM means each chip
   holds a smaller shard, reducing per-chip IO during weight streaming from the HF cache.

3. **Inference dominates wall time** (95–99%). Setup, exec, and test are negligible at this
   scale — optimising them would have no meaningful impact on throughput.

4. **Runaway tasks are the main efficiency bottleneck.** A small number of tasks consumed
   disproportionate wall time:
   - `find_in_sorted`: 2201s / 2308s for 1 real step — format error loop persists despite fix.
   - `levenshtein`, `longest_common_subsequence`, `mergesort`, `next_palindrome`,
     `max_sublist_sum`, `pascal`, `possible_change`: 300–580s/step — severe context explosion.
   These tasks alone account for most of the unfinished wall time budget.

5. **Neither 8h run completed all 40 tasks** (24/40 and 26/40). A full run of all 40 tasks
   requires roughly 10–12 GPU-hours at current throughput.

6. **Solve rate is ~37–42%** on completed tasks. The benchmark is non-trivial for this model.

---

### Lessons and next steps

#### What we learned

- The ROCm grouped-GEMM bottleneck means **more GPUs ≠ faster inference** for this model.
  The only benefit of 4 GPUs is faster model loading. For long batch jobs, 2 GPUs is the
  more efficient allocation (lower node cost, same throughput).
- **Context length is the real inference cost driver.** Capping the agent at fewer steps or
  summarising conversation history would reduce per-step cost more than adding GPUs.
- **Per-task wall time caps are essential.** A hard 20–30 min ceiling per task would prevent
  runaway loops from consuming the whole session budget without sacrificing normal tasks.
- **The pipeline works end-to-end** and produces clean, reproducible timing data at all phases.

#### If the experiment were to continue

- Add a per-task wall-time cap (e.g., 1800s) to prevent runaway tasks stealing budget.
- Investigate the persistent format error loop on `find_in_sorted` — the hard reminder fix
  helped most tasks but not this one.
- To complete all 40 tasks cleanly, submit a 3rd 8h run starting from task 25/27 onwards,
  or restructure as two back-to-back 8h jobs splitting the task list.
