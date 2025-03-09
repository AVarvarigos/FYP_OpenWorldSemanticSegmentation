/workspace/.local/conda/envs/openworld/bin/python train.py \
  --id someid --dataset_dir /workspace/ContMAV/datasets/cityscapes --num_classes 19 --batch_size 8 \
  --obj true \
  --mav true \
  --closs true \
  --loss_weights 1,1,1,1 \
  --workers 10 \
  --encoder resnet34 \
  --encoder_block NonBottleneck1D \
  --plot_results true \
  --freeze 20
  # --overfit true \
  # --debug
  # --no_imagenet_pretraining \

# /workspace/.local/conda/envs/openworld/bin/python train.py \
#   --id someId \
#   --dataset_dir /workspace/ContMAV/datasets/cityscapes \
#   --num_classes 19 \
#   --batch_size 10 \
#   --results_dir ./output \
#   --pretrained_dir /workspace/ContMAV/trained_models/imagenet \
#   --no_imagenet_pretraining \
#   --finetune None \
#   --freeze 0 \
#   --batch_size_valid 10 \
#   --width 1024 \
#   --epochs 1001 \
#   --lr 0.0001 \
#   --weight_decay 1e-4 \
#   --momentum 0.9 \
#   --optimizer Adam \
#   --class_weighting None \
#   --c_for_logarithmic_weighting 1.02 \
#   --he_init \
#   --valid_full_res \
#   --activation relu \
#   --encoder resnet34 \
#   --encoder_block NonBottleneck1D \
#   --nr_decoder_blocks 3 \
#   --modality rgb \
#   --encoder_decoder_fusion add \
#   --context_module appm-1-2-4-8 \
#   --channels_decoder 128 \
#   --decoder_channels_mode decreasing \
#   --upsampling learned-3x3-zeropad \
#   --aug_scale_min 0.5 \
#   --aug_scale_max 2.0 \
#   --plot_results true \
#   --obj true \
#   --mav true \
#   --closs true \
#   --loss_weights 1,1,1,1 \
#   --workers 10 \
#   --overfit true \
#   --debug
#   #   --last_ckpt "" \
#   # --load_weights "" \