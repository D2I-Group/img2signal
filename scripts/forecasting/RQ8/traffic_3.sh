export ckpt="./ckpt/"
export vm_arch="mae_base"
export context_len=1152
export align_const=0.4
export norm_const=0.4
export ft_type="ln"

# for PRED_LEN in 96 192 336 720; do
#     CUDA_VISIBLE_DEVICES=5 python -u run.py \
#     --task_name long_term_forecast \
#     --learning_rate 0.0001 \
#     --batch_size 512 \
#     --is_training 1 \
#     --model VisionTS \
#     --root_path ./dataset/traffic/ \
#     --data_path traffic.csv \
#     --save_dir save/traffic_3_$PRED_LEN \
#     --model_id VisionTS_traffic_$PRED_LEN \
#     --data custom \
#     --features M \
#     --train_epochs 1 \
#     --ckpt $ckpt \
#     --vm_arch $vm_arch \
#     --seq_len $context_len \
#     --periodicity 32 \
#     --pred_len $PRED_LEN \
#     --align_const $align_const \
#     --norm_const $norm_const \
#     --ft_type $ft_type
# done;



# for PRED_LEN in 96 192 336 720; do
#     CUDA_VISIBLE_DEVICES=5 python -u run.py \
#     --task_name long_term_forecast \
#     --learning_rate 0.0001 \
#     --batch_size 512 \
#     --is_training 1 \
#     --model VisionTS \
#     --root_path ./dataset/traffic/ \
#     --data_path traffic.csv \
#     --save_dir save/traffic_3_$PRED_LEN \
#     --model_id VisionTS_traffic_$PRED_LEN \
#     --data custom \
#     --features M \
#     --train_epochs 1 \
#     --ckpt $ckpt \
#     --vm_arch $vm_arch \
#     --seq_len $context_len \
#     --periodicity 36 \
#     --pred_len $PRED_LEN \
#     --align_const $align_const \
#     --norm_const $norm_const \
#     --ft_type $ft_type
# done;




for PRED_LEN in 96 192 336 720; do
    CUDA_VISIBLE_DEVICES=5 python -u run.py \
    --task_name long_term_forecast \
    --learning_rate 0.0001 \
    --batch_size 512 \
    --is_training 1 \
    --model VisionTS \
    --root_path ./dataset/traffic/ \
    --data_path traffic.csv \
    --save_dir save/traffic_3_$PRED_LEN \
    --model_id VisionTS_traffic_$PRED_LEN \
    --data custom \
    --features M \
    --train_epochs 1 \
    --ckpt $ckpt \
    --vm_arch $vm_arch \
    --seq_len $context_len \
    --periodicity 44 \
    --pred_len $PRED_LEN \
    --align_const $align_const \
    --norm_const $norm_const \
    --ft_type $ft_type
done;


for PRED_LEN in 96 192 336 720; do
    CUDA_VISIBLE_DEVICES=5 python -u run.py \
    --task_name long_term_forecast \
    --learning_rate 0.0001 \
    --batch_size 512 \
    --is_training 1 \
    --model VisionTS \
    --root_path ./dataset/traffic/ \
    --data_path traffic.csv \
    --save_dir save/traffic_3_$PRED_LEN \
    --model_id VisionTS_traffic_$PRED_LEN \
    --data custom \
    --features M \
    --train_epochs 1 \
    --ckpt $ckpt \
    --vm_arch $vm_arch \
    --seq_len $context_len \
    --periodicity 48 \
    --pred_len $PRED_LEN \
    --align_const $align_const \
    --norm_const $norm_const \
    --ft_type $ft_type
done;