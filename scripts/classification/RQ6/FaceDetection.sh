CUDA_VISIBLE_DEVICES=4 python -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path ./dataset/FaceDetection/ \
  --model_id FaceDetection \
  --model Linear_Cls_RQ5 \
  --data UEA \
  --batch_size 32 \
  --lradj fixed \
  --learning_rate 0.00001 \
  --lradj fixed \
  --train_epochs 0 \
  --time2img MVH \
  --resize default \
  --ft_type full \
  --patience 15 


CUDA_VISIBLE_DEVICES=4 python -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path ./dataset/FaceDetection/ \
  --model_id FaceDetection \
  --model Linear_Cls_RQ5 \
  --data UEA \
  --batch_size 32 \
  --lradj fixed \
  --learning_rate 0.00001 \
  --lradj fixed \
  --train_epochs 0 \
  --time2img MVH \
  --resize default \
  --ft_type full \
  --patience 15 \
  --shuffle_flag sf_all


CUDA_VISIBLE_DEVICES=4 python -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path ./dataset/FaceDetection/ \
  --model_id FaceDetection \
  --model Linear_Cls_RQ5 \
  --data UEA \
  --batch_size 32 \
  --lradj fixed \
  --learning_rate 0.00001 \
  --lradj fixed \
  --train_epochs 0 \
  --time2img MVH \
  --resize default \
  --ft_type full \
  --patience 15 \
  --shuffle_flag sf_half

CUDA_VISIBLE_DEVICES=4 python -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path ./dataset/FaceDetection/ \
  --model_id FaceDetection \
  --model Linear_Cls_RQ5 \
  --data UEA \
  --batch_size 32 \
  --lradj fixed \
  --learning_rate 0.00001 \
  --lradj fixed \
  --train_epochs 0 \
  --time2img MVH \
  --resize default \
  --ft_type full \
  --patience 15 \
  --shuffle_flag ex_half


CUDA_VISIBLE_DEVICES=4 python -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path ./dataset/FaceDetection/ \
  --model_id FaceDetection \
  --model Linear_Cls_RQ5 \
  --data UEA \
  --batch_size 32 \
  --lradj fixed \
  --learning_rate 0.00001 \
  --lradj fixed \
  --train_epochs 0 \
  --time2img MVH \
  --resize default \
  --ft_type full \
  --patience 15 \
  --shuffle_flag mask


CUDA_VISIBLE_DEVICES=4 python -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path ./dataset/FaceDetection/ \
  --model_id FaceDetection \
  --model Attn_Cls_RQ5 \
  --data UEA \
  --batch_size 32 \
  --lradj fixed \
  --learning_rate 0.00001 \
  --lradj fixed \
  --train_epochs 0 \
  --time2img MVH \
  --resize default \
  --ft_type full \
  --patience 15 


CUDA_VISIBLE_DEVICES=4 python -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path ./dataset/FaceDetection/ \
  --model_id FaceDetection \
  --model Attn_Cls_RQ5 \
  --data UEA \
  --batch_size 32 \
  --lradj fixed \
  --learning_rate 0.00001 \
  --lradj fixed \
  --train_epochs 0 \
  --time2img MVH \
  --resize default \
  --ft_type full \
  --patience 15 \
  --shuffle_flag sf_all


CUDA_VISIBLE_DEVICES=4 python -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path ./dataset/FaceDetection/ \
  --model_id FaceDetection \
  --model Attn_Cls_RQ5 \
  --data UEA \
  --batch_size 32 \
  --lradj fixed \
  --learning_rate 0.00001 \
  --lradj fixed \
  --train_epochs 0 \
  --time2img MVH \
  --resize default \
  --ft_type full \
  --patience 15 \
  --shuffle_flag sf_half


CUDA_VISIBLE_DEVICES=4 python -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path ./dataset/FaceDetection/ \
  --model_id FaceDetection \
  --model Attn_Cls_RQ5 \
  --data UEA \
  --batch_size 32 \
  --lradj fixed \
  --learning_rate 0.00001 \
  --lradj fixed \
  --train_epochs 0 \
  --time2img MVH \
  --resize default \
  --ft_type full \
  --patience 15 \
  --shuffle_flag ex_half


CUDA_VISIBLE_DEVICES=4 python -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path ./dataset/FaceDetection/ \
  --model_id FaceDetection \
  --model Attn_Cls_RQ5 \
  --data UEA \
  --batch_size 32 \
  --lradj fixed \
  --learning_rate 0.00001 \
  --lradj fixed \
  --train_epochs 0 \
  --time2img MVH \
  --resize default \
  --ft_type full \
  --patience 15 \
  --shuffle_flag mask