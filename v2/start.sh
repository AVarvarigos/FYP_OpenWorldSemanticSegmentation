/workspace/.local/envs/openworld/bin/python train.py \
  --id all --dataset_dir /workspace/FYP/FYP_OpenWorldSemanticSegmentation/v2/datasets/cityscapes --num_classes 19 --batch_size 8 \
  --pretrained_dir /workspace/Models/resnet34NonBottleneck1D \
<<<<<<< HEAD
  --class_weighting median_frequency \
  --obj true \
  --mav true \
  --closs true \
  --focal true \
=======
>>>>>>> main
  --loss_weights 1,1,1,1 \
  --workers 10 \
  --encoder resnet34 \
  --encoder_block NonBottleneck1D \
  --plot_results true\
<<<<<<< HEAD
  --lr 0.0005
=======
  --obj true \
  --lr 0.0005 \
  --overfit true \
  
  #--obj false\
  #--mav false\
  #--closs false \
>>>>>>> main
  # --overfit true \
  # --debug
  # --no_imagenet_pretraining \

# /workspace/.local/envs/openworld/bin/python train.py \
#   --id onlycrossentropy --dataset_dir /workspace/FYP/FYP_OpenWorldSemanticSegmentation/v2/datasets/cityscapes --num_classes 19 --batch_size 8 \
#   --pretrained_dir /workspace/Models/resnet34NonBottleneck1D \
#   --workers 10 \
#   --encoder resnet34 \
#   --encoder_block NonBottleneck1D \
#   --plot_results true \
#   --lr 0.004
