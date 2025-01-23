#!/bin/bash

# Enable logging
LOG_FILE="process_and_train.log"
exec > >(tee -a "$LOG_FILE") 2>&1
set -e  # Exit immediately if a command fails

echo "Process started at: $(date)"

# Base paths
BASE_DATA_PATH="/home/hk-project-test-p0021631/st_st190139/kge/data"
SOURCE_FOLDER="${BASE_DATA_PATH}/wn18"
KGE_CONFIG_PATH="/home/hk-project-test-p0021631/st_st190139/kge/examples/wn18-rescal.yaml"
CHECKPOINT_PATH="/home/hk-project-test-p0021631/st_st190139/kge/local/experiments/wn18-rescal/checkpoint_best.pt"

# Noise levels
NOISE_LEVELS=(0.01 0.05 0.1 0.2)

# Loop through noise levels
for NOISE_LEVEL in "${NOISE_LEVELS[@]}"; do
    # Create output folder
    OUTPUT_FOLDER="${BASE_DATA_PATH}/custom-dataset-${NOISE_LEVEL}"
    mkdir -p "${OUTPUT_FOLDER}"

    # Copy source data
    cp -r ${SOURCE_FOLDER}/* "${OUTPUT_FOLDER}"

    # Remove original splits
    rm -f "${OUTPUT_FOLDER}/train.txt" "${OUTPUT_FOLDER}/test.txt" "${OUTPUT_FOLDER}/valid.txt"

    # Generate custom datasets using Python
    python <<END
import os
import shutil
import yaml
import torch
import distribution_shift_utils as dsutils
from custom_distribution import CustomDistribution

output_folder = "${OUTPUT_FOLDER}"
noise_level = ${NOISE_LEVEL}
checkpoint_path = "${CHECKPOINT_PATH}"

# Generate custom datasets
custom_distribution = CustomDistribution(
    checkpoint_path=checkpoint_path,
    noise_magnitude=noise_level,
)

train_triples, valid_triples = custom_distribution.sample_training_validation_set()
original_test_ratio = len(custom_distribution.dataset.split("test")) / len(
    torch.cat((
        custom_distribution.dataset.split("train"),
        custom_distribution.dataset.split("valid"),
        custom_distribution.dataset.split("test")
    ))
)
test_triples, _ = custom_distribution.sample_test_set(
    torch.cat((custom_distribution.dataset.split("train"), custom_distribution.dataset.split("valid"))),
    original_test_ratio
)

# Save splits
dsutils.save_triples(train_triples, os.path.join(output_folder, "train.txt"))
dsutils.save_triples(valid_triples, os.path.join(output_folder, "valid.txt"))
dsutils.save_triples(test_triples, os.path.join(output_folder, "test.txt"))
END

    # Modify dataset.yaml
    DATASET_YAML_PATH="${OUTPUT_FOLDER}/dataset.yaml"
    python3 <<END
import yaml

output_folder = "${OUTPUT_FOLDER}"
noise_level = ${NOISE_LEVEL}
checkpoint_path = "${CHECKPOINT_PATH}"

yaml_path = "${DATASET_YAML_PATH}"
with open(yaml_path, 'r') as f:
    dataset_config = yaml.safe_load(f)

## Update flat keys
if 'dataset' in dataset_config:
    dataset = dataset_config['dataset']

    # Update file paths and sizes
    dataset['files.train.filename'] = 'train.txt'
    dataset['files.train.size'] = -1

    dataset['files.valid.filename'] = 'valid.txt'
    dataset['files.valid.size'] = -1

    dataset['files.test.filename'] = 'test.txt'
    dataset['files.test.size'] = -1

    # Update dataset name
    dataset['name'] = f"custom-dataset-{noise_level}"


# Write back the updated configuration to the file
with open(yaml_path, 'w') as f:
    yaml.safe_dump(dataset_config, f, default_flow_style=False)
END

    # Modify training configuration
    MODIFIED_KGE_CONFIG_PATH="${OUTPUT_FOLDER}/custom-rescal-${NOISE_LEVEL}.yaml"
    python3 <<END
import yaml

kge_config_path = "${KGE_CONFIG_PATH}"
output_path = "${MODIFIED_KGE_CONFIG_PATH}"
with open(kge_config_path, 'r') as f:
    kge_config = yaml.safe_load(f)

kge_config['dataset']['name'] = "custom-dataset-${NOISE_LEVEL}"

with open(output_path, 'w') as f:
    yaml.safe_dump(kge_config, f)
END

    # Start training
    kge start "${MODIFIED_KGE_CONFIG_PATH}"
done

echo "All datasets processed and models trained."

