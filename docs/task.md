# Topic: Efficient and Scalable Systems for Agentic AI
#### Supervisor: Zheyue Tan and Bo Zhao

* Background: Agentic AI represents the next paradigm of large-scale intelligent systems, where models act autonomously across multi-step reasoning, planning, and tool-use workflows. Unlike conventional LLM training and inference, agentic AI requires dynamic decision-making loops, inter-agent communication, and continuous interaction with external environments (e.g., APIs, databases, simulators). These characteristics introduce new challenges in system design, specifically:
    * Orchestration Overhead: Managing complex dependency graphs (DAGs) of agents.t
    * State Management: Efficiently handling long-context history and KV-cache across multi-turn sessions.
    * Heterogeneous Workloads: Synchronizing diverse resources (LLMs for reasoning, smaller models for routing or value estimation, vector DBs for memory) on distributed clusters.
* Tasks: Investigate state-of-the-art agentic system architectures, deploy them on CSC GPU clusters. Profile the system and seek potentional optimization opportunities.

## References:
OpenRLHF: An Easy-to-use, Scalable and High-performance RLHF Framework, arXiv, 2024, https://arxiv.org/pdf/2405.11143
HybridFlow: A Flexible and Efficient RLHF Framework, EuroSys, 2025, https://arxiv.org/pdf/2409.19256
RLHFuse: Efficient RLHF Training for Large Language Models with Inter- and Intra-Stage Fusion, arXiv, 2024, https://www.arxiv.org/pdf/2409.13221
ReaLHF: Optimized RLHF Training for Large Language Models through Parameter Reallocation, arXiv, 2024, https://arxiv.org/pdf/2406.14088
An Adaptive Placement and Parallelism Framework for Accelerating RLHF Training, arXiv, 2023, https://arxiv.org/pdf/2312.11819
Puzzle: Efficiently Aligning Large Language Models through Light-Weight Context Switch, USENIX ATC, 2024, https://www.usenix.org/system/files/atc24-lei.pdf
Murakkab: Resource-Efficient Agentic Workflow Orchestration in Cloud Platforms, arXiv, 2025, https://arxiv.org/pdf/2508.18298
Towards a Science of Scaling Agent Systems, arXiv, 2025, https://arxiv.org/abs/2512.08296v1
