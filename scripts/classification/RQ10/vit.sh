# CUDA_VISIBLE_DEVICES=5 python -u run.py \
#   --task_name classification \
#   --is_training 1 \
#   --root_path ./dataset/SpokenArabicDigits/ \
#   --model_id SpokenArabicDigits \
#   --model UTSClassification \
#   --data UEA \
#   --batch_size 16 \
#   --learning_rate 0.0001 \
#   --train_epochs 1 \
#   --time2img GAF \
#   --backbone vit_base \
#   --resize default \
#   --patience 3


# CUDA_VISIBLE_DEVICES=5 python -u run.py \
#   --task_name classification \
#   --is_training 1 \
#   --root_path ./dataset/Handwriting/ \
#   --model_id Handwriting \
#   --model UTSClassification \
#   --data UEA \
#   --batch_size 32 \
#   --learning_rate 0.0001 \
#   --train_epochs 1 \
#   --time2img GAF \
#   --lradj fixed \
#   --backbone vit_base \
#   --resize default \
#   --patience 10


CUDA_VISIBLE_DEVICES=5 python -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path ./dataset/FaceDetection/ \
  --model_id FaceDetection \
  --model UTSClassification \
  --data UEA \
  --batch_size 1 \
  --learning_rate 0.0001 \
  --train_epochs 1 \
  --time2img GAF \
  --backbone vit_base \
  --resize default \
  --patience 3


CUDA_VISIBLE_DEVICES=5 python -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path ./dataset/UWaveGestureLibrary/ \
  --model_id UWaveGestureLibrary \
  --model UTSClassification \
  --data UEA \
  --batch_size 32 \
  --learning_rate 0.0001 \
  --train_epochs 1 \
  --time2img GAF \
  --backbone vit_base \
  --resize default \
  --patience 3

