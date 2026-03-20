"""
Generate tasks.jsonl for experiment 4 from SWE-bench Lite.

Usage:
    # Specific instances:
    python3 generate_tasks.py --ids astropy__astropy-14365 django__django-11001

    # First N instances from the dataset:
    python3 generate_tasks.py --n 5
"""

import argparse
import json
from datasets import load_dataset

IMAGE_TEMPLATE = "ghcr.io/epoch-research/swe-bench.eval.x86_64.{instance_id}:latest"


def parse_list_field(value):
    if isinstance(value, list):
        return value
    if isinstance(value, str) and value:
        return json.loads(value)
    return []


def to_task(inst):
    fail_to_pass = parse_list_field(inst.get("FAIL_TO_PASS", []))
    pass_to_pass = parse_list_field(inst.get("PASS_TO_PASS", []))
    all_tests = " ".join(fail_to_pass + pass_to_pass)
    test_command = f"python -m pytest {all_tests} --no-header -rN -x -q 2>&1" if all_tests else ""
    return {
        "id":           inst["instance_id"],
        "description":  inst["problem_statement"],
        "docker_image": IMAGE_TEMPLATE.format(instance_id=inst["instance_id"]),
        "repo":         inst["repo"],
        "base_commit":  inst["base_commit"],
        "test_command": test_command,
        "fail_to_pass": fail_to_pass,
        "pass_to_pass": pass_to_pass,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ids", nargs="+", help="Specific instance IDs to include")
    parser.add_argument("--n", type=int, default=0, help="Take first N instances instead")
    parser.add_argument("--out", default="tasks.jsonl")
    parser.add_argument("--split", default="test")
    args = parser.parse_args()

    print("Loading SWE-bench Lite...")
    ds = load_dataset("princeton-nlp/SWE-bench_Lite", split=args.split)

    if args.ids:
        id_set = set(args.ids)
        instances = [inst for inst in ds if inst["instance_id"] in id_set]
        missing = id_set - {inst["instance_id"] for inst in instances}
        if missing:
            print(f"Warning: instance IDs not found: {missing}")
    elif args.n > 0:
        instances = list(ds)[:args.n]
    else:
        raise SystemExit("Provide --ids or --n")

    with open(args.out, "w") as f:
        for inst in instances:
            f.write(json.dumps(to_task(inst)) + "\n")

    print(f"Wrote {len(instances)} tasks to {args.out}")
    for inst in instances:
        ftp = parse_list_field(inst.get("FAIL_TO_PASS", []))
        print(f"  {inst['instance_id']:<45} fail_to_pass: {len(ftp)} tests")
    print(f"\nPull images with:")
    for inst in instances:
        print(f"  docker pull {IMAGE_TEMPLATE.format(instance_id=inst['instance_id'])}")


if __name__ == "__main__":
    main()
