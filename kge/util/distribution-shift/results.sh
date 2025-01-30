#!/bin/bash

# Enable logging
LOG_FILE="results.log"
exec > >(tee -a "$LOG_FILE") 2>&1
set -e

module load devel/cuda/12.4
source ~/miniconda3/bin/activate newkge
export PYTHONPATH=/home/hk-project-test-p0021631/st_st190139/miniconda3/envs/newkge/lib/python3.9/site-packages:$PYTHONPATH

# Define base paths and folders
BASE_DIR="/hkfs/home/project/hk-project-test-p0021631/st_st190139/kge/local/experiments"
RESULTS_DIR="/hkfs/home/project/hk-project-test-p0021631/st_st190139/kge/kge/util/distribution-shift/results/fb15k/distmult"
CHECKPOINT_FOLDERS=(
    "${BASE_DIR}/fb15k-distmult-0.01"
    "${BASE_DIR}/fb15k-distmult-0.05"
    "${BASE_DIR}/fb15k-distmult-0.1"
    "${BASE_DIR}/fb15k-distmult-0.2"
)

# Loop through the checkpoint folders and noise levels
for CHECKPOINT_FOLDER in "${CHECKPOINT_FOLDERS[@]}"; do
    # Extract noise level from the checkpoint folder name
    if [[ "${CHECKPOINT_FOLDER}" =~ fb15k-distmult-([0-9\.]+) ]]; then
        CHECKPOINT_NOISE_LEVEL="${BASH_REMATCH[1]}"
    else
        echo "Skipping non-matching folder: ${CHECKPOINT_FOLDER}"
        continue
    fi

    # Find the corresponding noise-level directory in results
    NOISE_DIR="${RESULTS_DIR}/${CHECKPOINT_NOISE_LEVEL}"

    # Create the noise-level directory if it doesn't exist
    mkdir -p "${NOISE_DIR}"

    # Define the log file paths for each script
    SUBGROUP_LOG="${NOISE_DIR}/subgroup-log.txt"
    DISTRIBUTION_LOG="${NOISE_DIR}/distribution-analysis-log.txt"

    # Run the subgroup.py script and log the output
    echo "Running Subgroup Evaluation for ${CHECKPOINT_FOLDER}..." | tee -a "${SUBGROUP_LOG}"
    python /hkfs/home/project/hk-project-test-p0021631/st_st190139/kge/kge/util/distribution-shift/subgroup.py \
        --checkpoint_path "${CHECKPOINT_FOLDER}/checkpoint_best.pt" \
        --group_type "relation" | tee -a "${SUBGROUP_LOG}"

    # Run the distribution analysis script and log the output
    echo "Running Distribution Analysis for ${CHECKPOINT_FOLDER}..." | tee -a "${DISTRIBUTION_LOG}"
    python /hkfs/home/project/hk-project-test-p0021631/st_st190139/kge/kge/util/distribution-shift/distribution_analysis.py \
        --checkpoint_path "${CHECKPOINT_FOLDER}/checkpoint_best.pt" | tee -a "${DISTRIBUTION_LOG}"
done

echo "Script execution completed."

