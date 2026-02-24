import os
import re
import subprocess
import argparse
import json
import shutil
import time
from datetime import datetime, timezone


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

## Overview

You're a software engineer interacting continuously with a computer by submitting commands.
You'll be helping implement necessary changes to meet requirements in the PR description.
Your task is specifically to make changes to non-test files in the current directory in order to fix the issue described in the PR description in a way that is general and consistent with the codebase.

<IMPORTANT>This is an interactive process where you will think and issue ONE command, see its result, then think and issue your next command.</IMPORTANT>

For each response:

1. Include a THOUGHT section explaining your reasoning and what you're trying to accomplish
2. Provide exactly ONE bash command to execute

## Important Boundaries

- MODIFY: Regular source code files in the working directory
- DO NOT MODIFY: Tests, configuration files (pyproject.toml, setup.cfg, etc.)

## Recommended Workflow

1. Analyze the codebase by finding and reading relevant files
2. Create a script to reproduce the issue
3. Edit the source code to resolve the issue
4. Verify your fix works by running your script again
5. Test edge cases to ensure your fix is robust

## Command Execution Rules

You are operating in an environment where

1. You write a single command
2. The system executes that command in a subshell with working directory: {repo_path}
3. You see the result
4. You write your next command

Each response should include:

1. A **THOUGHT** section where you explain your reasoning and plan
2. A single bash code block with your command

Format your responses like demonstrated within the <format_example> block:

<format_example>
THOUGHT: Here I explain my reasoning process, analysis of the current situation,
and what I'm trying to accomplish with the command below.

```mswea_bash_command
your_command_here
```
</format_example>

**CRITICAL REQUIREMENTS:**

- Your response SHOULD include a THOUGHT section explaining your reasoning
- Your response MUST include EXACTLY ONE bash code block
- This bash block MUST contain EXACTLY ONE command (or a set of commands connected with && or ||)
- If you include zero or multiple bash blocks, or no command at all, YOUR RESPONSE WILL FAIL

## Submission

When you've completed your work, you MUST submit your changes as a git patch.
Follow these steps IN ORDER, with SEPARATE commands:

Step 1: Create the patch file
Run `git diff -- path/to/file1 path/to/file2 > patch.txt` listing only the source files you modified.
Do NOT commit your changes.

Step 2: Verify your patch
Inspect patch.txt to confirm it only contains your intended changes and headers show `--- a/` and `+++ b/` paths.

Step 3: Submit (EXACT command required)
You MUST use this EXACT command to submit:

```mswea_bash_command
echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cat patch.txt
```

If the command fails (nonzero exit status), it will not submit.
</instructions>"""

FORMAT_ERROR_TEMPLATE = """\
Format error:

<error>
{error}
</error>

Please always provide EXACTLY ONE action in triple backticks.
Please format your action in triple backticks as shown in <response_example>.

<response_example>
Here are some thoughts about why you want to perform the action.

```mswea_bash_command
<action>
```
</response_example>"""

SUBMIT_MARKER = "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"


def load_tasks(path):
    tasks = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            tasks.append(json.loads(line))
    return tasks


def build_observation(returncode, output, exception_info=None):
    parts = []
    if exception_info:
        parts.append(f"<exception>{exception_info}</exception>")
    parts.append(f"<returncode>{returncode}</returncode>")
    if len(output) < 10000:
        parts.append(f"<output>\n{output}\n</output>")
    else:
        elided = len(output) - 10000
        parts.append(
            "<warning>\n"
            "The output of your last command was too long.\n"
            "Please try a different command that produces less output.\n"
            "</warning>"
        )
        parts.append(f"<output_head>\n{output[:5000]}\n</output_head>")
        parts.append(f"<elided_chars>\n{elided} characters elided\n</elided_chars>")
        parts.append(f"<output_tail>\n{output[-5000:]}\n</output_tail>")
    return "\n".join(parts)


def parse_command(text):
    match = re.search(r"```mswea_bash_command\n(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def run_command(command, cwd, timeout=60):
    try:
        result = subprocess.run(
            ["bash", "-c", command],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return result.returncode, result.stdout + result.stderr, None
    except subprocess.TimeoutExpired:
        return 1, "", "Command timed out"
    except Exception as e:
        return 1, "", str(e)


def run_tests(repo_root, command):
    result = subprocess.run(
        command,
        cwd=repo_root,
        shell=True,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.returncode == 0, result.stdout + result.stderr


def make_api_generator(client, model_name, max_new_tokens):
    """Returns a callable(messages) -> str using the OpenAI-compatible API."""
    def generate(messages):
        return client.chat.completions.create(
            model=model_name,
            messages=messages,
            max_tokens=max_new_tokens,
            temperature=0.0,
        ).choices[0].message.content or ""
    return generate


def make_local_generator(model, tokenizer, max_new_tokens, device):
    """Returns a callable(messages) -> str using a local HuggingFace model."""
    def generate(messages):
        inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = inputs.to(device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # greedy = temperature 0
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )
        generated_ids = outputs[0][inputs.input_ids.shape[1]:]
        return tokenizer.decode(generated_ids, skip_special_tokens=True)
    return generate


def run_task(task, generator, args):
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    work_root = os.path.join(args.work_dir, f"{task['id']}_{run_id}")
    repo_copy = os.path.join(work_root, "repo")
    os.makedirs(work_root, exist_ok=True)
    shutil.copytree(task["repo_path"], repo_copy)

    # Init git so the model can use `git diff` for submission
    subprocess.run(["git", "init"], cwd=repo_copy, capture_output=True)
    subprocess.run(["git", "config", "user.email", "agent@local"], cwd=repo_copy, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Agent"], cwd=repo_copy, capture_output=True)
    subprocess.run(["git", "add", "."], cwd=repo_copy, capture_output=True)
    subprocess.run(["git", "commit", "-m", "initial"], cwd=repo_copy, capture_output=True)

    log_path = os.path.join(work_root, "agent_log.jsonl")

    instance_prompt = INSTANCE_TEMPLATE.format(task=task["description"], repo_path=repo_copy)
    messages = [
        {"role": "system", "content": SYSTEM_TEMPLATE},
        {"role": "user", "content": instance_prompt},
    ]

    patch_content = None
    for step in range(args.step_limit):
        response = generator(messages)

        with open(log_path, "a") as f:
            f.write(json.dumps({"step": step + 1, "role": "assistant", "content": response}) + "\n")

        messages.append({"role": "assistant", "content": response})

        command = parse_command(response)
        if not command:
            n_blocks = len(re.findall(r"```mswea_bash_command", response))
            error = f"Please always provide EXACTLY ONE action in triple backticks, found {n_blocks} actions."
            obs = FORMAT_ERROR_TEMPLATE.format(error=error)
            messages.append({"role": "user", "content": obs})
            continue

        print(f"  Step {step + 1}: {command[:120]}")

        returncode, output, exception = run_command(command, repo_copy)

        with open(log_path, "a") as f:
            f.write(json.dumps({
                "step": step + 1, "role": "observation",
                "returncode": returncode, "output": output,
            }) + "\n")

        if SUBMIT_MARKER in output:
            patch_path = os.path.join(repo_copy, "patch.txt")
            if os.path.exists(patch_path):
                with open(patch_path) as f:
                    patch_content = f.read()
            else:
                after = output.split(SUBMIT_MARKER, 1)[1]
                patch_content = after.strip()
            print(f"  Submitted after {step + 1} steps.")
            patch_log = os.path.join(work_root, "submitted_patch.diff")
            with open(patch_log, "w") as f:
                f.write(patch_content)
            break

        obs = build_observation(returncode, output, exception)
        messages.append({"role": "user", "content": obs})
    else:
        print("  Step limit reached without submission.")

    tests_ok, test_log = run_tests(repo_copy, task["test_command"])
    return tests_ok, patch_content or "", test_log, work_root


def main():
    parser = argparse.ArgumentParser(description="Interactive agent harness using GLM (local GPU or API)")
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--model", default="zai-org/GLM-4.7-Flash")
    parser.add_argument("--max-new-tokens", type=int, default=1000)
    parser.add_argument("--step-limit", type=int, default=20)
    parser.add_argument("--work-dir", default="runs")
    parser.add_argument("--use-api", action="store_true",
                        help="Use HuggingFace inference API instead of local GPU")
    parser.add_argument("--api-base-url", default="https://router.huggingface.co/v1")
    args = parser.parse_args()

    if args.use_api:
        from openai import OpenAI
        hf_token = os.environ.get("HF_TOKEN")
        if not hf_token:
            raise SystemExit("HF_TOKEN environment variable is required for --use-api")
        client = OpenAI(base_url=args.api_base_url, api_key=hf_token)
        generator = make_api_generator(client, args.model, args.max_new_tokens)
        print(f"Mode: API ({args.api_base_url})")
    else:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Mode: local  Device: {device}")
        load_start = time.perf_counter()
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            trust_remote_code=True,
            torch_dtype=torch.float16,
        ).to(device)
        print(f"Model load time: {time.perf_counter() - load_start:.2f}s")
        generator = make_local_generator(model, tokenizer, args.max_new_tokens, device)

    tasks = load_tasks(args.tasks)
    if not tasks:
        raise SystemExit("No tasks found")
    tasks_dir = os.path.dirname(os.path.abspath(args.tasks))
    for task in tasks:
        repo_path = task.get("repo_path", "")
        if repo_path and not os.path.isabs(repo_path):
            task["repo_path"] = os.path.join(tasks_dir, repo_path)

    for task in tasks:
        print(f"\n=== Task {task['id']} ===")
        task_start = time.perf_counter()
        ok, patch, test_log, work_root = run_task(task, generator, args)
        task_seconds = time.perf_counter() - task_start
        print(f"Result: {'PASS' if ok else 'FAIL'}")
        print(f"Task time: {task_seconds:.2f}s")
        print(f"Work dir: {work_root}")
        if test_log:
            print("Test log:\n", test_log)


if __name__ == "__main__":
    main()
