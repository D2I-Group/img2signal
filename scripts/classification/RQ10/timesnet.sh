export CUDA_VISIBLE_DEVICES=1


# python -u run.py \
#   --task_name classification \
#   --is_training 1 \
#   --root_path ./dataset/SpokenArabicDigits/ \
#   --model_id SpokenArabicDigits \
#   --model TimesNet \
#   --data UEA \
#   --e_layers 2 \
#   --batch_size 16 \
#   --d_model 32 \
#   --d_ff 32 \
#   --top_k 2 \
#   --des 'Exp' \
#   --itr 1 \
#   --learning_rate 0.001 \
#   --train_epochs 1 \
#   --patience 3


# python run.py \
#     --task_name classification \
#     --is_training 1 \
#     --root_path ./dataset/Handwriting/ \
#     --model_id Handwriting \
#     --model TimesNet \
#     --data UEA \
#     --e_layers 2 \
#     --batch_size 16 \
#     --d_model 32 \
#     --d_ff 64 \
#     --top_k 3 \
#     --des 'Exp' \
#     --itr 1 \
#     --learning_rate 0.001 \
#     --train_epochs 1 \
#     --patience 3

python run.py \
  --task_name classification \
  --is_training 1 \
  --root_path ./dataset/UWaveGestureLibrary/ \
  --model_id UWaveGestureLibrary \
  --model TimesNet \
  --data UEA \
  --e_layers 2 \
  --batch_size 16 \
  --d_model 32 \
  --d_ff 64 \
  --top_k 3 \
  --des 'Exp' \
  --itr 1 \
  --learning_rate 0.001 \
  --train_epochs 30 \
  --patience 10

# python -u run.py \
#   --task_name classification \
#   --is_training 1 \
#   --root_path ./dataset/FaceDetection/ \
#   --model_id FaceDetection \
#   --model TimesNet \
#   --data UEA \
#   --e_layers 2 \
#   --batch_size 16 \
#   --d_model 64 \
#   --d_ff 256 \
#   --top_k 3 \
#   --num_kernels 4 \
#   --des 'Exp' \
#   --itr 1 \
#   --learning_rate 0.001 \
#   --train_epochs 30 \
#   --patience 10