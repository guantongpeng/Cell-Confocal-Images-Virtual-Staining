#!/bin/bash -x
# Provide the full path to the folder that stores the data downloaded from
# https://downloads.allencell.org/publication-data/label-free-prediction/index.html
# RAW_DATASET_DIR should have 13 sub-folders corresponding to 13 datasets.
RAW_DATASET_DIR="/data/DLdata/gtp/Nature_methods_dataset"
# Provide the dataset name.
DATASET=${1:-dna}
# Provide the GPU id. Use -1 for CPU only.
GPU_ID=${2:-4}
# Provide the name of your model. Use 'gvtnet_label-free_pretrained' for provided pretrained model.
MODEL_NAME=${3:-"gvtnet_label-free"}
# Provide the number of saved checkpoint. Use 'pretrained' for provided pretrained model.
CHECKPOINT_NUM=${4:-700}
# Provide the path to the main folder that saves transformed datasets, checkpoints and results.
SAVE_DIR="save_dir/${DATASET}"

CSV_DATASET_DIR="datasets/csvs/${DATASET}/"
TIFF_DATASET_DIR="${SAVE_DIR}/datasets/test"
RESULT_DIR="${SAVE_DIR}/results/${MODEL_NAME}"
MODEL_DIR="${SAVE_DIR}/models/${MODEL_NAME}"
NUM_TEST_PAIRS=20

# Pre-process the testing data and save them into the tiff format.
python datasets/generate_npz_or_tiff.py \
		--csv_dataset_dir ${CSV_DATASET_DIR} \
		--raw_dataset_dir ${RAW_DATASET_DIR} \
		--tiff_dataset_dir ${TIFF_DATASET_DIR} \
		--num_test_pairs ${NUM_TEST_PAIRS}

# Load the network configures according to MODEL_NAME	
cp network_configures/${MODEL_NAME}.py network_configure.py

# Predict using trained GVTNet
python predict.py \
		--gpu_id ${GPU_ID} \
		--tiff_dataset_dir ${TIFF_DATASET_DIR} \
		--num_test_pairs ${NUM_TEST_PAIRS} \
		--result_dir ${RESULT_DIR} \
		--model_dir ${MODEL_DIR} \
		--checkpoint_num ${CHECKPOINT_NUM}
		