# Notes from first meeting with TA and also session 30.1

## Starting plan (made with TA)
1. Supercomputer basics
   * How to use LUMI
   * Allocate resource
   * Queue jobs

2. Frameworks investigation & code hands on.
* Start with simple coding agent:
  * https://mini-swe-agent.com/latest/
  * https://github.com/SWE-bench/SWE-bench
  * https://www.tbench.ai/

3. Introduce basic profiling tools. Think up how to analyze results.
Overview - mini-SWE-agent documentation



## Progress first week
* LUMI access
* Running basic script in LUMI

## Learnt from session 30.1
* Possibility for opposition or teamwork with other project on same topic
* Importance of definging the term "scaling". Before I move on to the actual benchmarking, I need to consider what I here mean as scaling. Is it scaling n of nodes only? One agent or many? THis hsould be thought about, decided, visualized so it is clear

## Next goals
1. Look at docs of some simple model
2. Figure out what "scaling" means for said model. Visualize and define this (might change later as I move on)
3. Try simple benchmarks / measuring efficiency, resource use etc. Define also here what we actually want to look at / study

*** To figure out: how to take docker SWE benchmark and run in apptainer / on lumi


# Meeting 6.2

## Progress:
* Ran mini-SWE and SWE-Bench locally 


## Next: 
* How to scale
* Defining scope
* How to run
* Using models

* Running a model on LUMI: lumi has preconfigured pytorch container
  * Starting with [GLM 4.7 flash](https://huggingface.co/zai-org/GLM-4.7-Flash)

* Ask about collaboration

* Do it as a jupyter notebook job

* Goal: reduce end-2-end-latency for larger benchmark
