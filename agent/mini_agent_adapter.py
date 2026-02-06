# mini_agent_adapter.py
import subprocess
import json
from pathlib import Path

class MiniAgentAdapter:
    def __init__(self, model="default"):
        self.model = model
        self.output_dir = Path("results")
        self.output_dir.mkdir(exist_ok=True)

    def solve(self, task_text: str) -> str:
        """Called by SWE-bench for each task."""
        output_file = self.output_dir / "tmp.json"
        cmd = [
            "mini",
            "--task", task_text,
            "--model", self.model,
            "--yolo",
            "--output", str(output_file),
        ]
        subprocess.run(cmd, check=True)

        # read mini output
        if output_file.exists():
            with open(output_file, "r") as f:
                data = json.load(f)
            # here you extract the answer text from mini's JSON
            answer = data.get("final_answer", "")  # depends on mini output format
            return answer
        else:
            return ""
