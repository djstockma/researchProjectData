#!/bin/bash
# Complete pipeline to generate predictions and evaluate with SWE-bench

set -e  # Exit on error

echo "======================================"
echo "SWE-bench + mini-swe-agent Pipeline"
echo "======================================"
echo ""

# Step 1: Generate predictions
echo "Step 1: Generating predictions with mini-swe-agent..."
echo "This will take 10-30 minutes per task (currently MAX_TASKS=1)"
echo ""
python generate_mini_predictions.py

echo ""
echo "======================================"
echo "Step 1 Complete! Results saved to results/"
echo "======================================"
echo ""

# Step 2: Convert to SWE-bench format
echo "Step 2: Converting to SWE-bench JSONL format..."
python convert_to_swebench_format.py

echo ""
echo "======================================"
echo "Step 2 Complete! predictions.jsonl created"
echo "======================================"
echo ""

# Step 3: Run SWE-bench evaluation
echo "Step 3: Running SWE-bench evaluation..."
echo "This requires Docker and will take 5-15 minutes per task"
echo ""

cd swe-bench/SWE-bench

python -m swebench.harness.run_evaluation \
    --dataset_name princeton-nlp/SWE-bench_Lite \
    --predictions_path ../../predictions.jsonl \
    --max_workers 2 \
    --run_id mini_swe_eval_$(date +%Y%m%d_%H%M%S)

cd ../..

echo ""
echo "======================================"
echo "Pipeline Complete!"
echo "======================================"
echo ""
echo "Check results in: swe-bench/SWE-bench/evaluation_results/"
echo ""
echo "To view summary:"
echo "  cat swe-bench/SWE-bench/evaluation_results/mini_swe_eval_*/report.json"
