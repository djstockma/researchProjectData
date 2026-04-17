# Scaling AI Coding Agents on LUMI

![EuroHPC](https://img.shields.io/badge/EuroHPC-LUMI-0053A0?logo=linux&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![Model](https://img.shields.io/badge/LLM-GLM--4.7--Flash-blueviolet)
![Benchmark](https://img.shields.io/badge/Benchmark-QuixBugs-orange)

Course project by Jens Stockmarr, March 2026.

Can a locally-hosted open-weights LLM solve real software bugs autonomously — and how do the costs break down on a supercomputer GPU cluster? This project runs a mini SWE-agent loop using **GLM-4.7-Flash** (a 64 GB mixture-of-experts model) on [LUMI](https://www.lumi-supercomputer.eu/), Europe's fastest supercomputer, benchmarked against the **QuixBugs** suite of 40 Python algorithm bugs.

**Key findings:** inference dominates (95–99% of wall time), 4-GPU parallelism does *not* speed up inference due to a ROCm MoE routing fallback, and the agent achieves ~40% pass rate on QuixBugs.

## Report

See [`report/report.md`](report/report.md) for the full writeup.

## Structure

| Directory | Contents |
|-----------|----------|
| `experiments/lumi_glm_test_2/` | One-shot diff generation baseline |
| `experiments/lumi_glm_test_3/` | Interactive agent (mini-SWE-agent style), toy task |
| `experiments/lumi_glm_test_4/` | Interactive agent on real SWE-bench tasks |
| `experiments/lumi_glm_test_5/` | **Main experiment** — QuixBugs timing on LUMI |
| `report/` | Report, figures, and per-task result tables |
| `docs/` | Project notes and planning documents |
