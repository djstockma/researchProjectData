# Experiment 5 Design Notes: Why SWE-bench + Local GPU Doesn't Work

## The Problem

The goal was to run the full agent pipeline with a **local GPU model** (GLM-4.7-Flash)
on **SWE-bench tasks**, measuring per-phase timing. This turned out to be impossible
on LUMI due to a fundamental container nesting constraint.

## Why It Fails: Nested Singularity

On LUMI, the local GPU model requires the **LAIF container** (lumi-multitorch-latest.sif)
because:
- ROCm (AMD GPU drivers) is only available inside LAIF
- `transformers` with GLM-4.7-Flash support is only in LAIF/myvenv

SWE-bench tasks require **their own Singularity container** (one SIF per task) to provide
the correct Python environment and testbed for each task. The agent must call
`singularity exec <swe-sif> bash -c <command>` for every agent step.

**Running `singularity exec` from inside another `singularity exec` is not supported on LUMI.**
This is a known limitation of LUMI's Singularity setup — nested container invocations fail.

## What We Tried

| Approach | Result |
|----------|--------|
| Full sandbox extraction (`--swe-fs-dir`) to Lustre | Works but ~1h per task — too slow |
| Full sandbox extraction to `/tmp` (NVMe) before job | ~10-30 min per task; 10 tasks = up to 5h setup |
| Testbed-only extraction + `singularity exec` per step | Requires running outside LAIF → no GPU/ROCm |
| API mode outside LAIF + singularity exec per step | Works, but no GPU → no model load time measurement |

## What Works (but doesn't meet the goal)

- **GPU + local tasks (no containers):** Works, but toy tasks or similar are not a real benchmark.
- **API + SWE-bench (outside LAIF):** Works, but uses HF inference API — no model load time, no GPU.

## The Root Cause

LUMI's GPU software stack (ROCm) is only accessible inside the LAIF Singularity container.
SWE-bench task execution also requires a Singularity container per task. These two
requirements are mutually exclusive on LUMI: you cannot run `singularity exec` from
inside a `singularity exec`.

## Conclusion: Pivot to Container-Free Benchmark

To run with a **local GPU model** and measure all timing phases cleanly, we need tasks
that do **not require containers** — i.e., tasks that can run in plain Python directly
on the compute node.

The next step is to adopt a lightweight published benchmark (e.g. QuixBugs) that:
- Has real, citable tasks (not toy tasks we invented)
- Runs in plain Python with no Docker/Singularity overhead
- Allows the full pipeline (model load → inference → execution → test) to be timed cleanly
