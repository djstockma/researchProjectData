"""
Explore SWE-bench Lite: inspect task fields and save a sample tasks.jsonl
in the format expected by test_glm_agent.py.

Usage:
    python3 explore_swebench.py
    python3 explore_swebench.py --n 10 --out swe_tasks_sample.jsonl
"""

import argparse
import json
from datasets import load_dataset


FIELD_DESCRIPTIONS = {
    "instance_id":       "Unique task ID (e.g. django__django-11099)",
    "repo":              "GitHub repo (e.g. django/django)",
    "base_commit":       "Commit hash where the bug exists",
    "problem_statement": "The bug description shown to the agent",
    "hints_text":        "Optional hints (may be empty)",
    "test_cmd":          "Command to run tests",
    "FAIL_TO_PASS":      "Tests that must go from FAIL -> PASS (the fix target)",
    "PASS_TO_PASS":      "Tests that must stay passing (regression guard)",
}


def print_instance(idx, inst):
    print(f"\n{'='*70}")
    print(f"Instance {idx}: {inst['instance_id']}")
    print(f"{'='*70}")
    for field, desc in FIELD_DESCRIPTIONS.items():
        value = inst.get(field, "")
        if isinstance(value, list):
            value_str = json.dumps(value)
        else:
            value_str = str(value)
        # Truncate long fields for display
        if len(value_str) > 300:
            value_str = value_str[:300] + "... [truncated]"
        print(f"\n[{field}]  ({desc})")
        print(f"  {value_str}")


def parse_list_field(value):
    """FAIL_TO_PASS / PASS_TO_PASS are stored as JSON strings in the dataset."""
    if isinstance(value, list):
        return value
    if isinstance(value, str) and value:
        return json.loads(value)
    return []


def to_harness_task(inst):
    """Convert a SWE-bench instance to the tasks.jsonl format used by test_glm_agent.py."""
    fail_to_pass = parse_list_field(inst.get("FAIL_TO_PASS", []))
    pass_to_pass = parse_list_field(inst.get("PASS_TO_PASS", []))
    # Construct test command from the specific test IDs — SWE-bench doesn't store it directly
    all_tests = " ".join(fail_to_pass + pass_to_pass)
    test_command = f"pytest {all_tests} --no-header -rN -x -q" if all_tests else ""
    return {
        "id":           inst["instance_id"],
        "description":  inst["problem_statement"],
        "repo":         inst["repo"],
        "base_commit":  inst["base_commit"],
        "test_command": test_command,
        "fail_to_pass": fail_to_pass,
        "pass_to_pass": pass_to_pass,
        # repo_path is left blank — will be set up via git clone + apptainer
        "repo_path":    "",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n",   type=int, default=5,
                        help="Number of instances to inspect and save")
    parser.add_argument("--out", default="swe_tasks_sample.jsonl",
                        help="Output file for sample tasks")
    parser.add_argument("--split", default="test",
                        help="Dataset split to use (test / dev)")
    args = parser.parse_args()

    print(f"Loading SWE-bench Lite ({args.split} split)...")
    ds = load_dataset("princeton-nlp/SWE-bench_Lite", split=args.split)
    print(f"Total instances: {len(ds)}")
    print(f"Fields in dataset: {list(ds[0].keys())}")

    # Print field overview for first N instances
    for i in range(min(args.n, len(ds))):
        print_instance(i, ds[i])

    # Save sample in harness format
    with open(args.out, "w") as f:
        for i in range(min(args.n, len(ds))):
            task = to_harness_task(ds[i])
            f.write(json.dumps(task) + "\n")

    print(f"\n\nSaved {min(args.n, len(ds))} tasks to {args.out}")
    print("\nField summary across sampled instances:")
    print(f"  {'instance_id':<40} {'repo':<25} {'fail->pass':>10} {'pass->pass':>10}")
    print(f"  {'-'*40} {'-'*25} {'-'*10} {'-'*10}")
    for i in range(min(args.n, len(ds))):
        inst = ds[i]
        ftp = parse_list_field(inst.get("FAIL_TO_PASS", []))
        ptp = parse_list_field(inst.get("PASS_TO_PASS", []))
        print(f"  {inst['instance_id']:<40} {inst['repo']:<25} {len(ftp):>10} {len(ptp):>10}")


if __name__ == "__main__":
    main()
