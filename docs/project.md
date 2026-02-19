# Scaling AI Coding Agents on LUMI

## 1. Research Question

**Primary question:**
How does scaling AI coding agents on a supercomputer (LUMI) affect performance, efficiency, and solution quality when evaluated using standardized software engineering benchmarks (e.g. SWE-BENCH)?

**Sub-questions:**

* What does *scaling* mean in practice for AI coding agents (models, agents, prompts, parallelism)?
* Which scaling axes provide meaningful gains (number of agent, resource allocation)?
* What bottlenecks emerge when moving from single-agent to multi-agent execution?

---

## 2. Scope: What “Scaling” Means
There are many possible dimensions for scaling:

* **Model execution**

  * Running large language models locally on LUMI GPUs
  * Comparing single-GPU vs multi-GPU inference (future)

* **Agent parallelism**

  * Running many independent coding agents in parallel
  * Each agent attempts to solve the same or different benchmark tasks

* **Evaluation throughput**

  * High-volume generation of patches
  * Batch evaluation via SWE-BENCH

Out of scope for this project:

* Training or fine-tuning models
* Reinforcement learning
* Human-in-the-loop evaluation

---

## 3. High-Level Architecture

Goal pipeline:

```
[ LLM Model on LUMI ]
        ↓↑
[ (Fake → Real) Coding Agent ]
        ↓
[ Patch / Diff Artifact ]
        ↓
[ SWE-BENCH Evaluation ]
        ↓
[ Metrics + Logs ]
```

Key design principles:

* The **model** is treated as a black-box text generator. Ran on lumi, so its resource needs can ba studied in parallel with the agent
* The **agent** is responsible for framing prompts and interpreting outputs
* The **benchmark** only consumes clean, reproducible artifacts (patches)
---

## 4. Plan and progress

### Current Progress

* Local experiments with SWE-BENCH-style agent pipeline. Built an end-to-end pipeline using API model:
  * Agent generates solution
  * Patch is converted to SWE-BENCH format
  * SWE-BENCH evaluates correctness
  * Identified agent–benchmark impedance mismatch and the need for glue code

***NEW:***
* Running lightweight model (GLM 4.7 flash) on LUMI, GPU allocation

### WIP
* (Clarified research direction and goal)
* Try model with agent prompts. Alternatively, run model + agent?
* Check the format of output, make sure it generates "working" patches

### Next steps
* Use SWE-BENCH-style task descriptions as prompts. Study:
  * Output format
  * Diff-like behavior
  * Runtime and GPU utilization
* Run official SWE-BENCH evaluation
* Establish a baseline for local inference vs API-based models

---

## 5. OPtional long-term directions

Potential extensions (not committed yet):

* Multi-agent sampling per task
* Throughput-oriented batching strategies
* Studying diminishing returns of scaling
* Running controlled scaling experiments using SLURM job arrays

---

## 6. Current questions / Blockers



---

## 7. Guiding Principles
* Document decisions and dead ends
* Aim for simple but working pipelines

---

*This document is expected to evolve continuously as the project progresses.*
