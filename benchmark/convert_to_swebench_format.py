#!/usr/bin/env python3
"""
Convert mini-swe-agent outputs to SWE-bench evaluation format.

mini-swe-agent outputs individual JSON files per task.
SWE-bench expects a JSONL file with format:
{
  "instance_id": "repo__repo-issue",
  "model_name_or_path": "model-name", 
  "model_patch": "diff content"
}
"""
import json
import subprocess
from pathlib import Path

# Config
RESULTS_DIR = Path("results")
OUTPUT_FILE = Path("predictions.jsonl")
MODEL_NAME = "mini-swe-agent-gpt-4o-mini"

def generate_diff_from_workspace(instance_id):
    """
    Generate a git diff from the workspace where mini-swe-agent made changes.
    This assumes mini-swe-agent cloned the repo and made edits.
    """
    # Mini-swe-agent works in current directory, check if there are any modified files
    # For astropy task, check the astropy directory
    workspace_dir = Path(".")
    
    # Try to generate a diff using git if available
    try:
        result = subprocess.run(
            ["git", "diff", "--no-index", "/dev/null", "astropy/modeling/separable.py"],
            capture_output=True,
            text=True,
            cwd=workspace_dir
        )
        if result.stdout:
            # Clean up the diff to make it look like a proper git patch
            diff_output = result.stdout
            # Replace /dev/null with proper file paths
            diff_output = diff_output.replace("/dev/null", "a/astropy/modeling/separable.py")
            diff_output = diff_output.replace("astropy/modeling/separable.py", "b/astropy/modeling/separable.py")
            return diff_output
    except:
        pass
    
    # Fallback: try to read the file and create a simple patch
    try:
        sep_file = Path("astropy/modeling/separable.py")
        if sep_file.exists():
            content = sep_file.read_text()
            # Create a simple patch format
            patch = f"diff --git a/astropy/modeling/separable.py b/astropy/modeling/separable.py\n"
            patch += "new file mode 100644\n"
            patch += f"--- /dev/null\n"
            patch += f"+++ b/astropy/modeling/separable.py\n"
            patch += "@@ -0,0 +1,{} @@\n".format(len(content.splitlines()))
            for line in content.splitlines():
                patch += f"+{line}\n"
            return patch
    except:
        pass
    
    return ""

def extract_patch_from_mini_output(mini_json):
    """
    Extract the patch/diff from mini-swe-agent output.
    For mini-swe-agent v2.0, we need to generate a diff from the file changes.
    """
    # Check common patch locations first
    if "model_patch" in mini_json:
        return mini_json["model_patch"]
    
    if "patch" in mini_json:
        return mini_json["patch"]
    
    # For mini-swe-agent v2.0: check if submission field has content
    if "info" in mini_json and "submission" in mini_json["info"]:
        submission = mini_json["info"]["submission"]
        if submission and submission.strip():
            return submission
    
    # Try to extract from trajectory messages - look for file edits
    if "messages" in mini_json:
        # Look through messages for file content that was created/edited
        patch_lines = []
        for msg in mini_json["messages"]:
            if msg.get("role") == "tool" and "content" in msg:
                try:
                    import json
                    content_data = json.loads(msg["content"])
                    if "output" in content_data:
                        output = content_data["output"]
                        # Check if this looks like file content (has code-like patterns)
                        if any(keyword in output for keyword in ["def ", "class ", "import ", "from "]):
                            # This might be a file that was viewed/created
                            # We'll note it but continue looking for actual diffs
                            pass
                except:
                    pass
        
    # If no patch found, return empty string (will need manual extraction)
    print(f"Warning: No patch found in mini output - may need to generate diff from workspace")
    return ""

def main():
    predictions = []
    
    # Find all mini output files
    json_files = list(RESULTS_DIR.glob("*.json"))
    
    if not json_files:
        print(f"No JSON files found in {RESULTS_DIR}")
        return
    
    print(f"Found {len(json_files)} result files")
    
    for json_file in json_files:
        # Extract instance_id from filename
        instance_id = json_file.stem  # e.g., "astropy__astropy-12907"
        
        try:
            with open(json_file, 'r') as f:
                mini_output = json.load(f)
            
            # Extract patch from mini output
            patch = extract_patch_from_mini_output(mini_output)
            
            # If no patch found in JSON, try to generate from workspace
            if not patch or not patch.strip():
                print(f"  Attempting to generate diff from workspace for {instance_id}...")
                patch = generate_diff_from_workspace(instance_id)
            
            # Create SWE-bench format entry
            prediction = {
                "instance_id": instance_id,
                "model_name_or_path": MODEL_NAME,
                "model_patch": patch
            }
            
            predictions.append(prediction)
            if patch and patch.strip():
                print(f"✓ Processed {instance_id} ({len(patch)} chars)")
            else:
                print(f"⚠ Processed {instance_id} (no patch found)")
            
        except Exception as e:
            print(f"✗ Error processing {json_file}: {e}")
            continue
    
    # Write to JSONL
    with open(OUTPUT_FILE, 'w') as f:
        for pred in predictions:
            f.write(json.dumps(pred) + '\n')
    
    print(f"\nConverted {len(predictions)} predictions to {OUTPUT_FILE}")
    print(f"\nNext step: Run SWE-bench evaluation with:")
    print(f"  python -m swebench.harness.run_evaluation \\")
    print(f"    --dataset_name princeton-nlp/SWE-bench_Lite \\")
    print(f"    --predictions_path {OUTPUT_FILE} \\")
    print(f"    --max_workers 4 \\")
    print(f"    --run_id mini_swe_eval")

if __name__ == "__main__":
    main()
