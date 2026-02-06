#!/usr/bin/env python3
import json
import subprocess
from pathlib import Path
from datasets import load_dataset

# ----------------------------
# Config
# ----------------------------
DATASET_NAME = "princeton-nlp/SWE-bench_Lite"
OUTPUT_DIR = Path("results")
OUTPUT_DIR.mkdir(exist_ok=True)

MINI_CLI = "mini"  # Ensure 'mini' is in PATH
MINI_MODEL = "gpt-4o-mini"  # OpenAI's cost-effective model (was "openai/gpt-5-mini" which doesn't exist)
MAX_TASKS = 1  # Set to an int to limit number of tasks (for testing)

# ----------------------------
# Load SWE-bench Lite dataset
# ----------------------------
dataset = load_dataset(DATASET_NAME, split="test")

if MAX_TASKS:
    dataset = dataset.select(range(MAX_TASKS))

print(f"Loaded {len(dataset)} tasks from {DATASET_NAME}")

# ----------------------------
# Iterate over tasks and run mini agent
# ----------------------------
for idx, item in enumerate(dataset):
    task_id = item["instance_id"]  # SWE-bench Lite uses 'instance_id'
    task_text = item["problem_statement"]  # The task/problem statement

    print(f"[{idx+1}/{len(dataset)}] Running mini-SWE-agent on task {task_id}")

    # Output file for this task
    output_file = OUTPUT_DIR / f"{task_id}.json"

    # Run mini CLI
    cmd = [
        MINI_CLI,
        "--task", task_text,
        "--model", MINI_MODEL,
        "--yolo",  # run without confirmation
        "--output", str(output_file),
    ]

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running mini on task {task_id}: {e}")
        continue

    # Optional: check if mini produced output
    if not output_file.exists():
        print(f"Warning: No output for task {task_id}")

print(f"\nAll predictions saved to {OUTPUT_DIR}/")
