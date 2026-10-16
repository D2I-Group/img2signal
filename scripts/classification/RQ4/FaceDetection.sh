CUDA_VISIBLE_DEVICES=7 python -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path ./dataset/FaceDetection/ \
  --model_id FaceDetection \
  --model UTSClassification \
  --data UEA \
  --batch_size 1 \
  --learning_rate 0.0001 \
  --train_epochs 30 \
  --time2img GAF \
  --backbone vit_base \
  --resize default \
  --ft_type full \
  --patience 15

CUDA_VISIBLE_DEVICES=7 python -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path ./dataset/FaceDetection/ \
  --model_id FaceDetection \
  --model UTSClassification \
  --data UEA \
  --batch_size 1 \
  --learning_rate 0.0001 \
  --train_epochs 30 \
  --time2img GAF \
  --backbone vit_base \
  --resize default \
  --ft_type full_wo_cls_mask \
  --patience 15

CUDA_VISIBLE_DEVICES=7 python -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path ./dataset/FaceDetection/ \
  --model_id FaceDetection \
  --model UTSClassification \
  --data UEA \
  --batch_size 1 \
  --learning_rate 0.0001 \
  --train_epochs 30 \
  --time2img GAF \
  --backbone vit_base \
  --resize default \
  --ft_type mlp+ln \
  --patience 15


CUDA_VISIBLE_DEVICES=7 python -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path ./dataset/FaceDetection/ \
  --model_id FaceDetection \
  --model UTSClassification \
  --data UEA \
  --batch_size 1 \
  --learning_rate 0.0001 \
  --train_epochs 30 \
  --time2img GAF \
  --backbone vit_base \
  --resize default \
  --ft_type ln \
  --patience 15

CUDA_VISIBLE_DEVICES=7 python -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path ./dataset/FaceDetection/ \
  --model_id FaceDetection \
  --model UTSClassification \
  --data UEA \
  --batch_size 1 \
  --learning_rate 0.0001 \
  --train_epochs 30 \
  --time2img GAF \
  --backbone vit_base \
  --resize default \
  --ft_type zero-shot \
  --patience 15

CUDA_VISIBLE_DEVICES=7 python -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path ./dataset/FaceDetection/ \
  --model_id FaceDetection \
  --model UTSClassification \
  --data UEA \
  --batch_size 1 \
  --learning_rate 0.0001 \
  --train_epochs 30 \
  --time2img GAF \
  --backbone vit_base \
  --resize default \
  --ft_type full \
  --no_ckpt \
  --patience 15