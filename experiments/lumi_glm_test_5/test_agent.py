"""
Experiment 5: Interactive agent with detailed pipeline timing.

Measures timing for each pipeline phase:
  - model_load_time_s  (GPU mode only, session-level)
  - setup_time_s       (copy testbed, per task)
  - model_time_total_s (sum of all LLM inference calls, per task)
  - exec_time_total_s  (sum of all command executions, per task)
  - test_time_s        (final test run, per task)
  - total_wall_time_s  (full task wall time)

Two execution backends:
  - Local (default): tasks run directly in a temp copy of the task repo.
    Use with simple tasks (task format needs repo_path).
  - Singularity (--singularity-sif-dir): copies /testbed from SWE-bench SIF,
    runs each agent command via singularity exec. Use with SWE-bench tasks.

Usage:
    # API mode, local tasks
    HF_TOKEN=... python3 test_agent.py --tasks tasks.jsonl --use-api

    # API mode, SWE-bench tasks via Singularity (LUMI, outside LAIF)
    HF_TOKEN=... python3 test_agent.py --tasks tasks_swe.jsonl --use-api \\
        --singularity-sif-dir /scratch/.../sif_images

    # Local GPU mode
    python3 test_agent.py --tasks tasks.jsonl --model zai-org/GLM-4.7-Flash
"""

import os
import re
import subprocess
import argparse
import json
import shutil
import time
from datetime import datetime, timezone


# --- Prompt templates ---------------------------------------------------

SYSTEM_TEMPLATE = """\
You are a helpful assistant that can interact multiple times with a computer shell to solve programming tasks.
Your response must contain exactly ONE bash code block with ONE command (or commands connected with && or ||).

Include a THOUGHT section before your command where you explain your reasoning process.
Format your response as shown in <format_example>.

<format_example>
THOUGHT: Your reasoning and analysis here

```mswea_bash_command
your_command_here
```
</format_example>

Failure to follow these rules will cause your response to be rejected."""

INSTANCE_TEMPLATE = """\
<task_description>
{task}
</task_description>

<instructions>
# Task Instructions

You're a software engineer fixing a single-line bug in a QuixBugs Python program.
Your working directory is the QuixBugs repo root.
- Source files are in `python_programs/`
- Tests are in `python_testcases/`

<IMPORTANT>Issue ONE command per response. Wait for the result before continuing.</IMPORTANT>

## Recommended Workflow
1. Read the buggy source file: `cat python_programs/<name>.py`
2. Edit the source file to fix the bug
3. Run the tests to verify: `python3 -m pytest python_testcases/test_<name>.py -v`
4. Submit your fix

## Submission
When done, submit IN ORDER with SEPARATE commands:

Step 1: `git diff > patch.txt`
Step 2: Verify: `cat patch.txt`
Step 3: Submit:

```mswea_bash_command
echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cat patch.txt
```

## Rules
- MODIFY only the file in `python_programs/` — do NOT edit test files
- ONE bash code block per response, ONE command
- The QuixBugs repo root is your current working directory
</instructions>"""

FORMAT_ERROR_TEMPLATE = """\
Format error:

<error>
{error}
</error>

Provide EXACTLY ONE bash code block:

```mswea_bash_command
your_command_here
```"""

SUBMIT_MARKER = "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"


# --- Singularity execution (SWE-bench tasks) ----------------------------

# Conda activation prefix used inside SWE-bench containers
CONDA_PREFIX = (
    "source /opt/miniconda3/etc/profile.d/conda.sh && "
    "conda activate testbed && "
    "cd /testbed && "
)

SIF_FILENAME_TEMPLATE = "swe-bench.eval.x86_64.{instance_id}_latest.sif"


def get_sif_path(instance_id, sif_dir):
    return os.path.join(sif_dir, SIF_FILENAME_TEMPLATE.format(instance_id=instance_id))


def setup_singularity_workdir(sif_path, work_root, timeout=300):
    """Copy /testbed from SIF into work_root/testbed/ for writable access.

    Returns (testbed_dir, setup_time_s).
    """
    t0 = time.perf_counter()
    testbed_dir = os.path.join(work_root, "testbed")
    os.makedirs(testbed_dir, exist_ok=True)
    result = subprocess.run(
        ["singularity", "exec", sif_path,
         "bash", "-c", f"cp -rp /testbed/. {testbed_dir}/"],
        capture_output=True, text=True, timeout=timeout,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Testbed copy from SIF failed:\n{result.stderr}")
    setup_time = time.perf_counter() - t0
    return testbed_dir, setup_time


def run_in_singularity(sif_path, testbed_dir, command, timeout=120):
    """Run a shell command in the SIF with testbed_dir bound to /testbed."""
    full_cmd = CONDA_PREFIX + command
    try:
        result = subprocess.run(
            ["singularity", "exec",
             "--bind", f"{testbed_dir}:/testbed",
             sif_path, "bash", "-c", full_cmd],
            capture_output=True, text=True, timeout=timeout,
        )
        return result.returncode, result.stdout + result.stderr, None
    except subprocess.TimeoutExpired:
        return 1, "", "Command timed out"
    except Exception as e:
        return 1, "", str(e)


# --- Local execution ----------------------------------------------------

def setup_workdir(repo_path, work_root):
    """Copy task repo into work_root/repo/ and initialise a git repo.

    Returns (work_dir, setup_time_s).
    """
    t0 = time.perf_counter()
    dest = os.path.join(work_root, "repo")
    shutil.copytree(repo_path, dest)

    git_env = {
        **os.environ,
        "GIT_AUTHOR_NAME": "agent",
        "GIT_AUTHOR_EMAIL": "agent@experiment",
        "GIT_COMMITTER_NAME": "agent",
        "GIT_COMMITTER_EMAIL": "agent@experiment",
        "GIT_CONFIG_NOSYSTEM": "1",
    }
    subprocess.run(["git", "init"], cwd=dest, capture_output=True, env=git_env)
    subprocess.run(["git", "add", "-A"], cwd=dest, capture_output=True, env=git_env)
    subprocess.run(
        ["git", "-c", "commit.gpgsign=false", "commit", "-m", "initial"],
        cwd=dest, capture_output=True, env=git_env,
    )
    setup_time = time.perf_counter() - t0
    return dest, setup_time


def run_locally(work_dir, command, timeout=120):
    """Run a bash command in work_dir. Returns (returncode, output, exception_info)."""
    try:
        result = subprocess.run(
            ["bash", "-c", command],
            capture_output=True, text=True, timeout=timeout,
            cwd=work_dir,
        )
        return result.returncode, result.stdout + result.stderr, None
    except subprocess.TimeoutExpired:
        return 1, "", "Command timed out"
    except Exception as e:
        return 1, "", str(e)


# --- Agent utilities ----------------------------------------------------

def parse_command(text):
    match = re.search(r"```mswea_bash_command\n(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def build_observation(returncode, output, exception_info=None, steps_remaining=None):
    parts = []
    if steps_remaining is not None:
        if steps_remaining <= 3:
            parts.append(
                f"<warning>Only {steps_remaining} steps remaining. Submit NOW.</warning>"
            )
        else:
            parts.append(f"<steps_remaining>{steps_remaining}</steps_remaining>")
    if exception_info:
        parts.append(f"<exception>{exception_info}</exception>")
    parts.append(f"<returncode>{returncode}</returncode>")
    if len(output) < 5000:
        parts.append(f"<output>\n{output}\n</output>")
    else:
        elided = len(output) - 5000
        parts.append("<warning>Output too long. Use head/tail/grep for targeted output.</warning>")
        parts.append(f"<output_head>\n{output[:2500]}\n</output_head>")
        parts.append(f"<elided_chars>\n{elided} characters elided\n</elided_chars>")
        parts.append(f"<output_tail>\n{output[-2500:]}\n</output_tail>")
    return "\n".join(parts)


# --- Generators ---------------------------------------------------------

class APICreditsExhausted(Exception):
    pass


def make_api_generator(client, model_name, max_new_tokens):
    def generate(messages):
        try:
            return (
                client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    max_tokens=max_new_tokens,
                    temperature=0.0,
                )
                .choices[0]
                .message.content
                or ""
            )
        except Exception as e:
            if "402" in str(e) or "depleted" in str(e).lower() or "credits" in str(e).lower():
                raise APICreditsExhausted(str(e)) from e
            raise

    return generate


def make_local_generator(model, tokenizer, max_new_tokens, device):
    def generate(messages):
        inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )
        generated_ids = outputs[0][inputs.input_ids.shape[1]:]
        return tokenizer.decode(generated_ids, skip_special_tokens=True)

    return generate


# --- Task runner --------------------------------------------------------

def load_tasks(path):
    tasks = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                tasks.append(json.loads(line))
    return tasks


def run_task(task, generator, args, model_load_time=None):
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    work_root = os.path.join(args.work_dir, f"{task['id']}_{run_id}")
    os.makedirs(work_root, exist_ok=True)

    log_path = os.path.join(work_root, "agent_log.jsonl")
    metrics = {
        "task_id": task["id"],
        "start_time": datetime.now(timezone.utc).isoformat(),
        "model_load_time_s": model_load_time,   # None for API mode
        "setup_time_s": 0.0,
        "model_time_total_s": 0.0,
        "exec_time_total_s": 0.0,
        "test_time_s": 0.0,
        "total_wall_time_s": 0.0,
        "steps": 0,
        "submitted": False,
        "tests_passed": None,
        "step_log": [],
    }

    task_start = time.perf_counter()

    # --- Setup execution backend ---
    if args.singularity_sif_dir:
        sif_path = get_sif_path(task["id"], args.singularity_sif_dir)
        print(f"  Copying testbed from: {os.path.basename(sif_path)}")
        try:
            work_dir, setup_time = setup_singularity_workdir(sif_path, work_root)
        except Exception as e:
            print(f"  Singularity setup failed: {e}")
            return False, "", "Singularity setup failed", work_root, metrics
        exec_fn = lambda cmd, timeout=120: run_in_singularity(sif_path, work_dir, cmd, timeout)
    else:
        repo_path = task["repo_path"]
        if not os.path.isabs(repo_path):
            repo_path = os.path.join(os.path.dirname(os.path.abspath(args.tasks)), repo_path)
        try:
            work_dir, setup_time = setup_workdir(repo_path, work_root)
        except Exception as e:
            print(f"  Setup failed: {e}")
            return False, "", "Setup failed", work_root, metrics
        exec_fn = lambda cmd, timeout=120: run_locally(work_dir, cmd, timeout)

    metrics["setup_time_s"] = setup_time
    print(f"  Setup: {setup_time:.2f}s")

    instance_prompt = INSTANCE_TEMPLATE.format(task=task["description"])
    messages = [
        {"role": "system", "content": SYSTEM_TEMPLATE},
        {"role": "user", "content": instance_prompt},
    ]

    patch_content = ""
    consecutive_format_errors = 0
    try:
        for step in range(args.step_limit):
            # Model inference
            t0 = time.perf_counter()
            response = generator(messages)  # may raise APICreditsExhausted
            model_time = time.perf_counter() - t0
            metrics["model_time_total_s"] += model_time

            with open(log_path, "a") as f:
                f.write(
                    json.dumps(
                        {"step": step + 1, "role": "assistant",
                         "content": response, "model_time_s": model_time}
                    ) + "\n"
                )

            messages.append({"role": "assistant", "content": response})

            command = parse_command(response)
            if not command:
                consecutive_format_errors += 1
                n = len(re.findall(r"```mswea_bash_command", response))
                error_msg = f"Found {n} bash blocks, expected exactly 1."
                if consecutive_format_errors >= 2:
                    error_msg += "\n\nCRITICAL: You MUST wrap your command in exactly one ```mswea_bash_command block. Do not explain, do not use any other format. Just output the block."
                obs = FORMAT_ERROR_TEMPLATE.format(error=error_msg)
                messages.append({"role": "user", "content": obs})
                continue
            consecutive_format_errors = 0

            print(f"  Step {step + 1} [{model_time:.1f}s model]: {command[:80]}")

            # Execute command
            t0 = time.perf_counter()
            returncode, output, exception = exec_fn(command)
            exec_time = time.perf_counter() - t0
            metrics["exec_time_total_s"] += exec_time
            metrics["steps"] = step + 1

            step_entry = {
                "step": step + 1,
                "command": command,
                "returncode": returncode,
                "output": output[:2000],
                "model_time_s": model_time,
                "exec_time_s": exec_time,
            }
            metrics["step_log"].append(step_entry)
            with open(log_path, "a") as f:
                f.write(json.dumps({"role": "observation", **step_entry}) + "\n")

            print(f"    exec {exec_time:.2f}s  rc={returncode}")

            if SUBMIT_MARKER in output:
                metrics["submitted"] = True
                patch_file = os.path.join(work_dir, "patch.txt")
                if os.path.exists(patch_file):
                    with open(patch_file) as f:
                        patch_content = f.read()
                if not patch_content:
                    _, diff_out, _ = exec_fn("git diff HEAD", timeout=30)
                    patch_content = diff_out
                    print("  (patch collected via git diff HEAD fallback)")
                if patch_content:
                    with open(os.path.join(work_root, "submitted_patch.diff"), "w") as f:
                        f.write(patch_content)
                print(f"  Submitted after {step + 1} steps.")
                break

            steps_remaining = args.step_limit - (step + 1)
            obs = build_observation(returncode, output, exception, steps_remaining=steps_remaining)
            messages.append({"role": "user", "content": obs})
        else:
            print("  Step limit reached without submission.")

        # Run tests
        if task.get("test_command"):
            t0 = time.perf_counter()
            _, test_output, _ = exec_fn(task["test_command"], timeout=30)
            metrics["test_time_s"] = time.perf_counter() - t0
            tests_passed = bool(re.search(r"\d+ passed", test_output)) and not re.search(
                r"\d+ failed|\d+ error", test_output
            )
            metrics["tests_passed"] = tests_passed
            with open(os.path.join(work_root, "test_output.txt"), "w") as f:
                f.write(test_output)
        else:
            test_output = ""
            tests_passed = False

    finally:
        pass  # work_root preserved for inspection

    metrics["total_wall_time_s"] = time.perf_counter() - task_start
    with open(os.path.join(work_root, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # Remove repo copy — metrics/logs/patch are sufficient; repo is just workspace
    shutil.rmtree(work_dir, ignore_errors=True)

    return metrics["tests_passed"] or False, patch_content, test_output, work_root, metrics


# --- Main ---------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Experiment 5: agent on simple local Python tasks with timing"
    )
    parser.add_argument("--tasks", required=True, help="Path to tasks.jsonl")
    parser.add_argument("--model", default="zai-org/GLM-4.7-Flash:novita",
                        help="Model name (HF ID for local, router ID for API)")
    parser.add_argument("--max-new-tokens", type=int, default=1000)
    parser.add_argument("--step-limit", type=int, default=15)
    parser.add_argument("--work-dir", default="runs")
    parser.add_argument("--use-api", action="store_true",
                        help="Use HuggingFace inference router instead of local model")
    parser.add_argument("--api-base-url", default="https://router.huggingface.co/v1")
    parser.add_argument("--singularity-sif-dir", default=None,
                        help="Directory containing SWE-bench .sif files. "
                             "If set, uses singularity exec for task commands (LUMI, outside LAIF).")
    args = parser.parse_args()

    model_load_time = None

    if args.use_api:
        from openai import OpenAI
        hf_token = os.environ.get("HF_TOKEN")
        if not hf_token:
            raise SystemExit("HF_TOKEN required for --use-api")
        client = OpenAI(base_url=args.api_base_url, api_key=hf_token)
        generator = make_api_generator(client, args.model, args.max_new_tokens)
        print(f"Mode: API  Model: {args.model}")
    else:
        import torch
        import transformers
        from transformers import AutoTokenizer, AutoModelForCausalLM

        transformers.logging.set_verbosity_error()
        transformers.logging.disable_progress_bar()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Mode: local  Device: {device}")

        t0 = time.perf_counter()
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.model, trust_remote_code=True, torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        model_load_time = time.perf_counter() - t0
        print(f"Model load time: {model_load_time:.2f}s")
        generator = make_local_generator(model, tokenizer, args.max_new_tokens, device)

    tasks = load_tasks(args.tasks)
    if not tasks:
        raise SystemExit("No tasks found")

    os.makedirs(args.work_dir, exist_ok=True)

    # Save session-level metadata (includes model load time for GPU mode)
    session_meta = {
        "model": args.model,
        "use_api": args.use_api,
        "model_load_time_s": model_load_time,
        "step_limit": args.step_limit,
        "max_new_tokens": args.max_new_tokens,
        "session_start": datetime.now(timezone.utc).isoformat(),
        "task_count": len(tasks),
    }
    with open(os.path.join(args.work_dir, "session_meta.json"), "w") as f:
        json.dump(session_meta, f, indent=2)

    summary = []
    try:
        for task in tasks:
            print(f"\n=== Task: {task['id']} ===")
            ok, patch, test_log, work_root, metrics = run_task(
                task, generator, args, model_load_time
            )
            result = "PASS" if ok else "FAIL"
            print(
                f"Result: {result}  "
                f"Steps: {metrics['steps']}  "
                f"Wall: {metrics['total_wall_time_s']:.1f}s  "
                f"Model: {metrics['model_time_total_s']:.1f}s  "
                f"Exec: {metrics['exec_time_total_s']:.1f}s  "
                f"Setup: {metrics['setup_time_s']:.2f}s  "
                f"Test: {metrics['test_time_s']:.2f}s"
            )
            summary.append({
                "id": task["id"],
                "result": result,
                **{k: metrics[k] for k in [
                    "steps", "submitted", "tests_passed",
                    "setup_time_s", "model_time_total_s", "exec_time_total_s",
                    "test_time_s", "total_wall_time_s",
                ]},
            })
    except APICreditsExhausted as e:
        print(f"\nStopped: API credits exhausted — {e}")

    passed = sum(1 for s in summary if s["result"] == "PASS")
    print(f"\n{'=' * 60}")
    print(f"Summary: {passed}/{len(summary)} passed")
    if model_load_time is not None:
        print(f"Model load time: {model_load_time:.2f}s")

    summary_data = {"session": session_meta, "tasks": summary}
    summary_path = os.path.join(args.work_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary_data, f, indent=2)
    print(f"Summary written to {summary_path}")


if __name__ == "__main__":
    main()
