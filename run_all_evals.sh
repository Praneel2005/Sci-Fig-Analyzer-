#!/bin/bash
# run_all_evals.sh
# Evaluates all 10 improved checkpoints against the test set.

OUTPUT_REPORT="improved_model_evaluation_report.txt"
echo "IMPROVED MODEL EVALUATION REPORT" > $OUTPUT_REPORT
echo "================================" >> $OUTPUT_REPORT
echo "" >> $OUTPUT_REPORT

for i in {1..10}
do
    echo "------------------------------------------------"
    echo "Evaluating Epoch $i..."
    echo "------------------------------------------------"
    
    # Run the test script (assuming test_blip.py exists)
    # Redirecting output to capture the final metrics
    conda run -n figcaps python test_blip.py \
        --model_path blip_improved_checkpoints/epoch_${i}/pytorch_model.bin \
        --benchmark_path /home/drive4/figcaps_data \
        2>&1 | tee eval_epoch_${i}.log
    
    # Extract metrics from log (assuming standard output format)
    echo "Epoch $i Metrics:" >> $OUTPUT_REPORT
    grep -E "BLEU|ROUGE|METEOR" eval_epoch_${i}.log >> $OUTPUT_REPORT
    echo "" >> $OUTPUT_REPORT
done

echo "Evaluation Complete. See $OUTPUT_REPORT for details."
