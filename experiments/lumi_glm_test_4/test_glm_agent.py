"""
Interactive agent harness — SWE-bench tasks in Docker (local) or Apptainer (LUMI) containers.

Local test with API model:
    HF_TOKEN=... python3 test_glm_agent.py --tasks tasks.jsonl --use-api

Local test with local model (needs GPU):
    python3 test_glm_agent.py --tasks tasks.jsonl

LUMI (local GPU + Apptainer):
    python3 test_glm_agent.py --tasks tasks.jsonl --singularity-sif-dir $SIF_DIR
    where SIF_DIR contains e.g. swe-bench.eval.x86_64.astropy__astropy-12907_latest.sif
    Pull SIFs on login node: singularity pull docker://ghcr.io/epoch-research/swe-bench.eval.x86_64.<id>:latest
"""

import os
import re
import subprocess
import argparse
import json
import shutil
import time
from datetime import datetime, timezone

# --- Prompt templates (same as experiment 3) ----------------------------

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
<pr_description>
Consider the following PR description:
{task}
</pr_description>

<instructions>
# Task Instructions

You're a software engineer interacting continuously with a computer shell to fix a bug.
Your working directory is /testbed (the repository root).

<IMPORTANT>Issue ONE command per response. Wait for the result before continuing.</IMPORTANT>

## Recommended Workflow
1. Read relevant source files to understand the bug
2. Edit the source code to fix it
3. Verify the fix by running the target tests
4. Submit a git patch

## Submission
When done, submit IN ORDER with SEPARATE commands:

Step 1: `git diff -- path/to/changed/file > patch.txt`
Step 2: Verify: `cat patch.txt`
Step 3: Submit:

```mswea_bash_command
echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cat patch.txt
```

## Rules
- MODIFY source files only — not tests or config
- ONE bash code block per response, ONE command
- Commands run in /testbed with the correct Python environment active
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

# Prefix that activates the right conda env inside the SWE-bench container
CONDA_PREFIX = (
    "source /opt/miniconda3/etc/profile.d/conda.sh && "
    "conda activate testbed && "
    "cd /testbed && "
)

# Image registry for SWE-bench instances (Epoch AI mirror)
IMAGE_TEMPLATE = "ghcr.io/epoch-research/swe-bench.eval.x86_64.{instance_id}:latest"

# Apptainer SIF filename produced by: singularity pull docker://<image>
SIF_FILENAME_TEMPLATE = "swe-bench.eval.x86_64.{instance_id}_latest.sif"


# --- Docker container lifecycle -----------------------------------------

def start_container(image):
    """Start a detached container with no network. Returns container ID."""
    result = subprocess.run(
        ["docker", "run", "-d", "--network", "none", image, "tail", "-f", "/dev/null"],
        capture_output=True, text=True, check=True,
    )
    return result.stdout.strip()


def stop_container(container_id):
    subprocess.run(["docker", "stop", "-t", "5", container_id], capture_output=True)
    subprocess.run(["docker", "rm", container_id], capture_output=True)


def run_in_container(container_id, command, timeout=120):
    """Run a shell command inside the container with conda env activated."""
    full_cmd = CONDA_PREFIX + command
    try:
        result = subprocess.run(
            ["docker", "exec", container_id, "bash", "-c", full_cmd],
            capture_output=True, text=True, timeout=timeout,
        )
        return result.returncode, result.stdout + result.stderr, None
    except subprocess.TimeoutExpired:
        return 1, "", "Command timed out"
    except Exception as e:
        return 1, "", str(e)


def copy_patch_from_container(container_id, dest_path):
    """Copy /testbed/patch.txt out of the container. Returns content or ''."""
    result = subprocess.run(
        ["docker", "cp", f"{container_id}:/testbed/patch.txt", dest_path],
        capture_output=True,
    )
    if result.returncode == 0 and os.path.exists(dest_path):
        with open(dest_path) as f:
            return f.read()
    return ""


# --- Apptainer container lifecycle (nested, currently broken on LUMI) ----

def get_sif_path(instance_id, sif_dir):
    return os.path.join(sif_dir, SIF_FILENAME_TEMPLATE.format(instance_id=instance_id))


def setup_singularity_workdir(sif_path, work_root, timeout=600):
    """Copy /testbed from the SIF image into work_root/testbed/ for writable access."""
    testbed_dir = os.path.join(work_root, "testbed")
    os.makedirs(testbed_dir, exist_ok=True)
    result = subprocess.run(
        ["singularity", "exec", "--userns",
         "--bind", f"{work_root}:/hostdir",
         sif_path, "bash", "-c", "cp -rp /testbed/. /hostdir/testbed/"],
        capture_output=True, text=True, timeout=timeout,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Apptainer testbed copy failed:\n{result.stderr}")
    return testbed_dir


def run_in_singularity(sif_path, testbed_dir, command, timeout=120):
    """Run a shell command in the singularity image with testbed_dir bound to /testbed."""
    full_cmd = CONDA_PREFIX + command
    try:
        result = subprocess.run(
            ["singularity", "exec", "--userns",
             "--bind", f"{testbed_dir}:/testbed",
             sif_path, "bash", "-c", full_cmd],
            capture_output=True, text=True, timeout=timeout,
        )
        return result.returncode, result.stdout + result.stderr, None
    except subprocess.TimeoutExpired:
        return 1, "", "Command timed out"
    except Exception as e:
        return 1, "", str(e)


# --- Extracted sandbox mode (no nested Singularity) ---------------------
# Usage: pre-extract SIF on login node with:
#   apptainer build --sandbox $SWE_FS $SWE_SIF
# Then pass --swe-fs-dir $SWE_FS_ROOT (parent dir, indexed by instance_id).

def get_swe_fs_path(instance_id, swe_fs_root):
    return os.path.join(swe_fs_root, instance_id)


def setup_extracted_workdir(swe_fs_dir, work_root):
    """Copy testbed from pre-extracted sandbox directory into work_root/testbed/ for writable access."""
    testbed_src = os.path.join(swe_fs_dir, "testbed")
    testbed_dir = os.path.join(work_root, "testbed")
    result = subprocess.run(
        ["cp", "-rp", testbed_src, testbed_dir],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Testbed copy from sandbox failed:\n{result.stderr}")
    return testbed_dir


def run_in_extracted(swe_fs_dir, testbed_dir, command, timeout=120):
    """Run a shell command using the pre-extracted SIF sandbox's conda environment."""
    conda_bin = os.path.join(swe_fs_dir, "opt/miniconda3/envs/testbed/bin")
    conda_base_bin = os.path.join(swe_fs_dir, "opt/miniconda3/bin")
    conda_lib = os.path.join(swe_fs_dir, "opt/miniconda3/envs/testbed/lib")

    env = os.environ.copy()
    env["PATH"] = f"{conda_bin}:{conda_base_bin}:{env.get('PATH', '')}"
    env["LD_LIBRARY_PATH"] = f"{conda_lib}:{env.get('LD_LIBRARY_PATH', '')}"
    env["CONDA_PREFIX"] = os.path.join(swe_fs_dir, "opt/miniconda3/envs/testbed")
    env["CONDA_DEFAULT_ENV"] = "testbed"
    env.pop("PYTHONPATH", None)  # prevent LAIF's Python 3.12 packages from leaking into testbed's Python 3.9
    env.pop("HOME", None)  # avoid conda searching /root which may not exist in LAIF

    full_cmd = f"cd {testbed_dir} && {command}"
    try:
        result = subprocess.run(
            ["bash", "-c", full_cmd],
            capture_output=True, text=True, timeout=timeout, env=env,
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
        if steps_remaining <= 5:
            parts.append(f"<warning>Only {steps_remaining} steps remaining. You MUST make a fix and submit soon.</warning>")
        else:
            parts.append(f"<steps_remaining>{steps_remaining}</steps_remaining>")
    if exception_info:
        parts.append(f"<exception>{exception_info}</exception>")
    parts.append(f"<returncode>{returncode}</returncode>")
    if len(output) < 10000:
        parts.append(f"<output>\n{output}\n</output>")
    else:
        elided = len(output) - 10000
        parts.append("<warning>Output too long. Use head/tail/grep for targeted output.</warning>")
        parts.append(f"<output_head>\n{output[:5000]}\n</output_head>")
        parts.append(f"<elided_chars>\n{elided} characters elided\n</elided_chars>")
        parts.append(f"<output_tail>\n{output[-5000:]}\n</output_tail>")
    return "\n".join(parts)


# --- Generators ---------------------------------------------------------

class APICreditsExhausted(Exception):
    pass


def make_api_generator(client, model_name, max_new_tokens):
    def generate(messages):
        try:
            return client.chat.completions.create(
                model=model_name,
                messages=messages,
                max_tokens=max_new_tokens,
                temperature=0.0,
            ).choices[0].message.content or ""
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


def run_task(task, generator, args):
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    work_root = os.path.join(args.work_dir, f"{task['id']}_{run_id}")
    os.makedirs(work_root, exist_ok=True)

    image = task.get("docker_image", IMAGE_TEMPLATE.format(instance_id=task["id"]))
    log_path = os.path.join(work_root, "agent_log.jsonl")
    metrics = {
        "task_id": task["id"],
        "docker_image": image,
        "start_time": datetime.now(timezone.utc).isoformat(),
        "model_time_total": 0.0,
        "exec_time_total": 0.0,
        "steps": 0,
        "submitted": False,
        "tests_passed": None,
        "step_log": [],
    }

    # --- Set up execution backend ---
    if args.swe_fs_dir:
        # Pre-extracted sandbox mode: no nested Singularity needed.
        swe_fs_dir = get_swe_fs_path(task["id"], args.swe_fs_dir)
        print(f"  Setting up workdir from extracted sandbox: {swe_fs_dir}")
        try:
            testbed_dir = setup_extracted_workdir(swe_fs_dir, work_root)
        except Exception as e:
            print(f"  Failed to set up workdir from sandbox: {e}")
            return False, "", "Sandbox setup failed", work_root, metrics

        def exec_fn(cmd, timeout=120):
            return run_in_extracted(swe_fs_dir, testbed_dir, cmd, timeout)

        def get_patch_fn(dest):
            patch_src = os.path.join(testbed_dir, "patch.txt")
            if os.path.exists(patch_src):
                shutil.copy(patch_src, dest)
                with open(dest) as f:
                    return f.read()
            return ""

        def teardown_fn():
            pass  # testbed stays in work_root for inspection

    elif args.singularity_sif_dir:
        sif_path = get_sif_path(task["id"], args.singularity_sif_dir)
        print(f"  Setting up singularity workdir: {os.path.basename(sif_path)}")
        try:
            testbed_dir = setup_singularity_workdir(sif_path, work_root)
        except Exception as e:
            print(f"  Failed to set up singularity workdir: {e}")
            return False, "", "Apptainer setup failed", work_root, metrics

        def exec_fn(cmd, timeout=120):
            return run_in_singularity(sif_path, testbed_dir, cmd, timeout)

        def get_patch_fn(dest):
            patch_src = os.path.join(testbed_dir, "patch.txt")
            if os.path.exists(patch_src):
                shutil.copy(patch_src, dest)
                with open(dest) as f:
                    return f.read()
            return ""

        def teardown_fn():
            pass  # testbed stays in work_root for inspection
    else:
        print(f"  Starting container: {image}")
        try:
            container_id = start_container(image)
        except subprocess.CalledProcessError as e:
            print(f"  Failed to start container: {e}")
            return False, "", "Container start failed", work_root, metrics

        def exec_fn(cmd, timeout=120):
            return run_in_container(container_id, cmd, timeout)

        def get_patch_fn(dest):
            return copy_patch_from_container(container_id, dest)

        def teardown_fn():
            stop_container(container_id)

    instance_prompt = INSTANCE_TEMPLATE.format(task=task["description"])
    messages = [
        {"role": "system", "content": SYSTEM_TEMPLATE},
        {"role": "user", "content": instance_prompt},
    ]

    patch_content = ""
    try:
        for step in range(args.step_limit):
            # Model inference
            t0 = time.perf_counter()
            response = generator(messages)   # raises APICreditsExhausted → propagates to main()
            model_time = time.perf_counter() - t0
            metrics["model_time_total"] += model_time

            with open(log_path, "a") as f:
                f.write(json.dumps({"step": step + 1, "role": "assistant",
                                    "content": response, "model_time": model_time}) + "\n")

            messages.append({"role": "assistant", "content": response})

            command = parse_command(response)
            if not command:
                n = len(re.findall(r"```mswea_bash_command", response))
                obs = FORMAT_ERROR_TEMPLATE.format(
                    error=f"Found {n} bash blocks, expected exactly 1.")
                messages.append({"role": "user", "content": obs})
                continue

            print(f"  Step {step + 1}: {command[:100]}")

            # Execute
            t0 = time.perf_counter()
            returncode, output, exception = exec_fn(command)
            exec_time = time.perf_counter() - t0
            metrics["exec_time_total"] += exec_time
            metrics["steps"] = step + 1

            step_entry = {
                "step": step + 1, "role": "observation",
                "command": command, "returncode": returncode,
                "output": output[:2000],   # truncate for log
                "model_time": model_time, "exec_time": exec_time,
            }
            with open(log_path, "a") as f:
                f.write(json.dumps(step_entry) + "\n")

            if SUBMIT_MARKER in output:
                metrics["submitted"] = True
                patch_content = get_patch_fn(os.path.join(work_root, "submitted_patch.diff"))
                if not patch_content:
                    # Fall back 1: extract diff from command output (agent did git diff && cat inline)
                    after = output.split(SUBMIT_MARKER, 1)[1]
                    candidate = after.strip()
                    if candidate.startswith("diff --git") or candidate.startswith("---"):
                        patch_content = candidate
                        with open(os.path.join(work_root, "submitted_patch.diff"), "w") as f:
                            f.write(patch_content)
                if not patch_content:
                    # Fall back 2: collect git diff directly (agent forgot to write patch.txt)
                    _, diff_out, _ = exec_fn("git diff HEAD", timeout=30)
                    if diff_out.strip().startswith("diff --git") or diff_out.strip().startswith("---"):
                        patch_content = diff_out
                        with open(os.path.join(work_root, "submitted_patch.diff"), "w") as f:
                            f.write(patch_content)
                        print("  (patch collected via git diff HEAD fallback)")
                print(f"  Submitted after {step + 1} steps.")
                break

            steps_remaining = args.step_limit - (step + 1)
            obs = build_observation(returncode, output, exception, steps_remaining=steps_remaining)
            messages.append({"role": "user", "content": obs})
        else:
            print("  Step limit reached without submission.")

        # Run tests to verify
        if task.get("test_command"):
            _, test_output, _ = exec_fn(task["test_command"], timeout=180)
            tests_passed = (
                bool(re.search(r"\d+ passed", test_output))
                and not re.search(r"\d+ failed", test_output)
                and not re.search(r"^(FAILED|ERROR)\b", test_output, re.MULTILINE)
            )
            metrics["tests_passed"] = tests_passed
            with open(os.path.join(work_root, "test_output.txt"), "w") as f:
                f.write(test_output)
        else:
            test_output = ""
            tests_passed = False

    finally:
        teardown_fn()

    metrics["total_wall_time"] = metrics["model_time_total"] + metrics["exec_time_total"]
    with open(os.path.join(work_root, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics["tests_passed"] or False, patch_content, test_output, work_root, metrics


# --- Main ---------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="SWE-bench agent harness with Docker containers")
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--model", default="zai-org/GLM-4.7-Flash:novita")
    parser.add_argument("--max-new-tokens", type=int, default=2000)
    parser.add_argument("--step-limit", type=int, default=20)
    parser.add_argument("--work-dir", default="runs")
    parser.add_argument("--use-api", action="store_true")
    parser.add_argument("--api-base-url", default="https://router.huggingface.co/v1")
    parser.add_argument("--singularity-sif-dir", default=None,
                        help="Directory containing .sif files for Apptainer (LUMI). "
                             "If omitted, Docker is used instead.")
    parser.add_argument("--swe-fs-dir", default=None,
                        help="Parent directory of pre-extracted SIF sandboxes (one subdir per instance_id). "
                             "Extract with: apptainer build --sandbox $SWE_FS_ROOT/<id> <id>.sif")
    args = parser.parse_args()

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
            args.model, trust_remote_code=True, torch_dtype=torch.float16,
        ).to(device)
        print(f"Model load time: {time.perf_counter() - t0:.2f}s")
        generator = make_local_generator(model, tokenizer, args.max_new_tokens, device)

    tasks = load_tasks(args.tasks)
    if not tasks:
        raise SystemExit("No tasks found")

    summary = []
    try:
        for task in tasks:
            print(f"\n=== Task {task['id']} ===")
            task_start = time.perf_counter()
            ok, patch, test_log, work_root, metrics = run_task(task, generator, args)
            wall = time.perf_counter() - task_start
            result = "PASS" if ok else "FAIL"
            print(f"Result: {result}  |  Steps: {metrics['steps']}  |  "
                  f"Wall: {wall:.1f}s  |  Model: {metrics['model_time_total']:.1f}s  |  "
                  f"Exec: {metrics['exec_time_total']:.1f}s")
            print(f"Work dir: {work_root}")
            summary.append({"id": task["id"], "result": result, "steps": metrics["steps"],
                            "wall_time": wall, **metrics})
    except APICreditsExhausted as e:
        print(f"\nStopped: API credits exhausted — {e}")

    print(f"\n{'='*60}")
    print(f"Summary: {sum(1 for s in summary if s['result'] == 'PASS')}/{len(summary)} passed")
    summary_path = os.path.join(args.work_dir, "summary.json")
    os.makedirs(args.work_dir, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary written to {summary_path}")


if __name__ == "__main__":
    main()
