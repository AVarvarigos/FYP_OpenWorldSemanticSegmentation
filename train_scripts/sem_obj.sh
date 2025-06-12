#!/bin/bash

source /workspace/FYP/FYP_OpenWorldSemanticSegmentation/train_scripts/vars.sh

export RUN_ID=sem_obj
echo "Job started on: $(date) with RUN_ID: $RUN_ID"

# Change to the project directory
cd /workspace/FYP/FYP_OpenWorldSemanticSegmentation/fyp
python train.py \
  --id $RUN_ID \
  --dataset_dir $DATASET_DIR \
  --pretrained_dir $PRETRAINED_DIR \
  --num_classes $NUM_CLASSES --batch_size $BATCH_SIZE \
  --lr $LR \
  --class_weighting median_frequency \
  --loss_weights 1,1,1,1 \
  --workers 10 \
  --encoder resnet34 \
  --encoder_block NonBottleneck1D \
  --plot_results true \
  --obj true

echo "Job completed on: $(date)"

