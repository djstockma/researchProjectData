#!/usr/bin/env python3
import json
import subprocess
from pathlib import Path

# ----------------------------
# Config
# ----------------------------
# Path to SWE-bench dataset (or any custom task list)
DATASET_NAME = "princeton-nlp/SWE-bench_Lite"
# Path to save agent outputs
OUTPUT_DIR = Path("results")
OUTPUT_DIR.mkdir(exist_ok=True)
# Model name to pass to mini
MINI_MODEL = "default"  # or specify a model class

# Path to mini CLI (ensure it's in PATH, otherwise give full path)
MINI_CLI = "mini"

# Number of tasks to run (None = all)
MAX_TASKS = None

# ----------------------------
# Example: fetch tasks from SWE-bench
# ----------------------------
# For demonstration, you can replace this with loading a JSON dataset
# Here we just use a placeholder list
tasks = [
    "Compute the integral of x^2",
    "Solve 2 + 2 * 3",
]

if MAX_TASKS:
    tasks = tasks[:MAX_TASKS]

# ----------------------------
# Run mini-SWE-agent on each task
# ----------------------------
results = []

for i, task in enumerate(tasks, start=1):
    print(f"[{i}/{len(tasks)}] Running mini-SWE-agent on task: {task}")
    # Output JSON file for this task
    output_file = OUTPUT_DIR / f"task_{i}.json"

    # Run mini CLI
    cmd = [
        MINI_CLI,
        "--task", task,
        "--model", MINI_MODEL,
        "--yolo",  # run without confirmation
        "--output", str(output_file),
    ]
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running task {i}: {e}")
        continue

    # Read mini output
    if output_file.exists():
        with open(output_file, "r") as f:
            data = json.load(f)
        results.append({
            "task": task,
            "mini_output": data,
        })
    else:
        print(f"No output file generated for task {i}")

# ----------------------------
# Save all results to a single JSON
# ----------------------------
final_file = OUTPUT_DIR / "all_results.json"
with open(final_file, "w") as f:
    json.dump(results, f, indent=2)

print(f"All tasks completed. Results saved to {final_file}")
