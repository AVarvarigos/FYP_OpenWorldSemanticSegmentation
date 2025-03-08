#!/bin/bash
#PBS -lselect=1:ncpus=16:mem=64gb:ngpus=1
#PBS -lwalltime=48:00:00

# cd $PBS_O_WORKDIR

# eval "$(~/anaconda3/bin/conda shell.bash hook)"
# source activate torch

export CUDA_LAUNCH_BLOCKING=1
python train.py \
  --id someId \
  --dataset_dir /teamspace/studios/this_studio/Open-World-Semantic-Segmentation-Including-Class-Similarity/datasets/cityscapes \
  --num_classes 19 \
  --batch_size 10 \
  --results_dir ./output \
  --last_ckpt "" \
  --load_weights "" \
  --pretrained_dir /teamspace/studios/this_studio/Open-World-Semantic-Segmentation-Including-Class-Similarity/trained_models/imagnet \
  --pretrained_on_imagenet \
  --finetune None \
  --freeze 0 \
  --batch_size_valid 10 \
  --width 1024 \
  --epochs 1001 \
  --lr 0.0001 \
  --weight_decay 1e-4 \
  --momentum 0.9 \
  --optimizer Adam \
  --class_weighting None \
  --c_for_logarithmic_weighting 1.02 \
  --he_init \
  --valid_full_res \
  --activation relu \
  --encoder resnet34 \
  --encoder_block NonBottleneck1D \
  --nr_decoder_blocks 3 \
  --modality rgb \
  --encoder_decoder_fusion add \
  --context_module appm-1-2-4-8 \
  --channels_decoder 128 \
  --decoder_channels_mode decreasing \
  --upsampling learned-3x3-zeropad \
  --aug_scale_min 0.5 \
  --aug_scale_max 2.0 \
  --obj \
  --mav \
  --closs \
  --loss_weights 1,1,1,1 \
  --workers 10
  # --overfit False \
  # --debug
