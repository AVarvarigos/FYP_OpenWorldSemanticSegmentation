/workspace/.local/envs/openworld/bin/python train.py \
  --id someid --dataset_dir /workspace/FYP/FYP_OpenWorldSemanticSegmentation/v2/datasets/cityscapes --num_classes 19 --batch_size 8 \
  --pretrained_dir /workspace/Models/resnet34NonBottleneck1D \
  --obj true \
  --mav true \
  --closs true \
  --loss_weights 1,1,1,1 \
  --workers 10 \
  --encoder resnet34 \
  --encoder_block NonBottleneck1D \
  --plot_results true \
  --lr 0.004
  # --overfit true \
  # --debug
  # --no_imagenet_pretraining \
