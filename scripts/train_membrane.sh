#!/bin/bash -x

DATASET="membrane_caax_63x"
# Provide the full path to the folder that stores the data downloaded from
# https://downloads.allencell.org/publication-data/label-free-prediction/index.html
# RAW_DATASET_DIR should have 13 sub-folders corresponding to 13 datasets.
RAW_DATASET_DIR="/data/DLdata/gtp/Nature_methods_dataset"
# Provide the GPU id. Use -1 for CPU only.
GPU_ID=${1:-4}
# Provide the name of your model.
MODEL_NAME=${2:-"gvtnet_label-free"}
# Provide the path to the main folder that saves transformed datasets, checkpoints and results.
SAVE_DIR="save_dir/${DATASET}"

CSV_DATASET_DIR="datasets/csvs/${DATASET}/"
NPZ_DATASET_DIR="${SAVE_DIR}/datasets/train"
RESULT_DIR="${SAVE_DIR}/results/${MODEL_NAME}"
MODEL_DIR="${SAVE_DIR}/models/${MODEL_NAME}"
NUM_TRAIN_PAIRS=30
BATCH_SIZE=16
LOSS_TYPE="MSE"
LEARNING_RATE=0.001
NUM_ITERS=100000
SAVE_CHECKPOINTS_ITER=5000
TRAIN_PATCH_SIZE_D=32
TRAIN_PATCH_SIZE_H=64
TRAIN_PATCH_SIZE_W=64

# Pre-process the training data and save them into the npz format.
python datasets/generate_npz_or_tiff.py \
		--csv_dataset_dir ${CSV_DATASET_DIR} \
		--raw_dataset_dir ${RAW_DATASET_DIR} \
		--npz_dataset_dir ${NPZ_DATASET_DIR} \
		--num_train_pairs ${NUM_TRAIN_PAIRS} \
		--transform_signal transforms.normalize "transforms.Resizer((1, 0.29655, 0.29655))" \
		--transform_target transforms.normalize "transforms.Resizer((1, 0.29655, 0.29655))"

# Build the network according to MODEL_NAME
# If network_configures/${MODEL_NAME}.py exists, copy it to network_configure.py
# If not, save the current network_configure.py to network_configures/${MODEL_NAME}.py
if [ -f network_configures/${MODEL_NAME}.py ]
then
	cp network_configures/${MODEL_NAME}.py network_configure.py
else
	cp network_configure.py network_configures/${MODEL_NAME}.py
fi

# Train the GVTNet
python train.py \
		--gpu_id ${GPU_ID} \
		--npz_dataset_dir ${NPZ_DATASET_DIR} \
		--train_patch_size ${TRAIN_PATCH_SIZE_D} ${TRAIN_PATCH_SIZE_H} ${TRAIN_PATCH_SIZE_W} \
		--num_train_pairs ${NUM_TRAIN_PAIRS} \
		--batch_size ${BATCH_SIZE} \
		--loss_type ${LOSS_TYPE} \
		--learning_rate ${LEARNING_RATE} \
		--num_iters ${NUM_ITERS} \
		--save_checkpoints_iter ${SAVE_CHECKPOINTS_ITER} \
		--model_dir ${MODEL_DIR}
