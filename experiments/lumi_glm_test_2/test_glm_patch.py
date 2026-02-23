import os

# Set environment variables before importing HF libraries to suppress progress bars
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import json
import shutil
import subprocess
import tempfile
import time
from datetime import datetime, timezone

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def read_file(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def load_tasks(path):
    tasks = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            task = json.loads(line)
            tasks.append(task)
    return tasks


def build_prompt(task, repo_root):
    file_blocks = []
    for rel_path in task["files"]:
        abs_path = os.path.join(repo_root, rel_path)
        content = read_file(abs_path)
        file_blocks.append(
            f"--- path: {rel_path}\n{content}\n--- end: {rel_path}\n"
        )

    files_text = "\n".join(file_blocks)
    instructions = (
        "You are an automated patch generator.\n"
        "Return a unified diff ONLY between PATCH_START and PATCH_END.\n"
        "Use standard unified diff headers (--- and +++).\n"
        "Do not include explanations, code fences, or extra text.\n"
        "Use real newlines, not escaped \\n sequences.\n"
        "Output format:\nPATCH_START\n<diff>\nPATCH_END\n"
    )
    example = (
        "Example:\n"
        "PATCH_START\n"
        "--- math_utils.py\n"
        "+++ math_utils.py\n"
        "@@ -1,3 +1,3 @@\n"
        "-return 1\n"
        "+return n\n"
        "PATCH_END\n"
    )
    prompt = (
        f"{instructions}\n"
        f"{example}\n"
        f"Task: {task['description']}\n\n"
        f"Files:\n{files_text}\n"
        "PATCH_START\n"
    )
    return prompt


def extract_patch(text):
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        cleaned = cleaned.replace("diff\n", "", 1)
    if "PATCH_START" in text and "PATCH_END" in text:
        return text.split("PATCH_START", 1)[1].split("PATCH_END", 1)[0].strip()
    if "diff --git" in cleaned:
        return cleaned.split("diff --git", 1)[1].strip()
    if cleaned.startswith("--- ") and "+++ " in cleaned:
        return cleaned
    return ""


def apply_patch(repo_root, patch_text):
    if not patch_text.strip():
        return False, "Empty patch"

    with tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8") as tmp:
        tmp.write(patch_text)
        tmp_path = tmp.name

    try:
        result = subprocess.run(
            ["patch", "-p0", "-i", tmp_path],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            return False, result.stderr + result.stdout
        return True, result.stdout
    finally:
        os.unlink(tmp_path)


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


def generate_patch_local(model, tokenizer, prompt, max_new_tokens, device):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.7,
        top_p=1.0,
        min_p=0.01,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return decoded


def generate_patch_api(client, model_name, prompt, max_new_tokens):
    completion = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_new_tokens,
        temperature=0.0,
        stop=["PATCH_END"],
    )
    return completion.choices[0].message.content or ""


def run_task(task, generator, args):
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    work_root = os.path.join(args.work_dir, f"{task['id']}_{run_id}")
    repo_copy = os.path.join(work_root, "repo")
    os.makedirs(work_root, exist_ok=True)

    shutil.copytree(task["repo_path"], repo_copy)

    attempt = 0
    last_error = ""
    while attempt <= args.max_retries:
        attempt += 1
        prompt = build_prompt(task, repo_copy)
        if last_error:
            prompt += (
                "\nPrevious attempt failed with:\n"
                f"{last_error}\n\n"
                "PATCH_START\n"
            )

        decoded = generator(prompt)
        output_path = os.path.join(work_root, f"model_output_attempt_{attempt}.txt")
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(decoded)
        patch_text = extract_patch(decoded)
        ok, apply_log = apply_patch(repo_copy, patch_text)
        if not ok:
            last_error = f"Patch apply failed:\n{apply_log}"
            continue

        tests_ok, test_log = run_tests(repo_copy, task["test_command"])
        if tests_ok:
            return True, apply_log, test_log, work_root

        last_error = f"Tests failed:\n{test_log}"

    return False, last_error, "", work_root


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--model", default="zai-org/GLM-4.7-Flash")
    parser.add_argument("--max-new-tokens", type=int, default=800)
    parser.add_argument("--max-retries", type=int, default=1)
    parser.add_argument("--work-dir", default="runs")
    parser.add_argument("--use-api", action="store_true")
    parser.add_argument("--api-base-url", default="https://router.huggingface.co/v1")
    args = parser.parse_args()

    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", args.device)

    generator = None
    if args.use_api:
        hf_token = os.environ.get("HF_TOKEN")
        if not hf_token:
            raise SystemExit("HF_TOKEN is required when using --use-api")
        from openai import OpenAI

        client = OpenAI(base_url=args.api_base_url, api_key=hf_token)

        def generator(prompt):
            return generate_patch_api(client, args.model, prompt, args.max_new_tokens)

    else:
        load_start = time.perf_counter()
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            trust_remote_code=True,
            torch_dtype=torch.float16,
        ).to(args.device)
        load_seconds = time.perf_counter() - load_start
        print(f"Model load time: {load_seconds:.2f}s")

        def generator(prompt):
            return generate_patch_local(
                model, tokenizer, prompt, args.max_new_tokens, args.device
            )

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
        ok, apply_log, test_log, work_root = run_task(task, generator, args)
        task_seconds = time.perf_counter() - task_start
        if ok:
            print("Result: PASS")
        else:
            print("Result: FAIL")
        print(f"Task time: {task_seconds:.2f}s")
        print("Work dir:", work_root)
        if apply_log:
            print("Apply log:\n", apply_log)
        if test_log:
            print("Test log:\n", test_log)


if __name__ == "__main__":
    main()
