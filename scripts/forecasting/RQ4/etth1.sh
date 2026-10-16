export ckpt="./ckpt/"
export vm_arch="mae_base"
export context_len=336
export periodicity=24
export align_const=0.4
export norm_const=0.4


for PRED_LEN in 96 192 336 720; do
    CUDA_VISIBLE_DEVICES=5 python -u run.py \
    --task_name long_term_forecast \
    --learning_rate 0.0001 \
    --batch_size 32 \
    --is_training 1 \
    --model VisionTS \
    --root_path ./dataset/ETT-small/ \
    --data_path ETTh1.csv \
    --save_dir save/ETTh1_RQ4_$PRED_LEN \
    --model_id VisionTS_ETTh1_$PRED_LEN \
    --data ETTh1 \
    --features M \
    --train_epochs 20 \
    --ckpt $ckpt \
    --lradj fixed \
    --vm_arch $vm_arch \
    --seq_len $context_len \
    --periodicity $periodicity \
    --pred_len $PRED_LEN \
    --align_const $align_const \
    --norm_const $norm_const \
    --ft_type ln
done;

# for PRED_LEN in 96 192 336 720; do
#     CUDA_VISIBLE_DEVICES=5 python -u run.py \
#     --task_name long_term_forecast \
#     --learning_rate 0.0001 \
#     --batch_size 32 \
#     --is_training 1 \
#     --model VisionTS \
#     --root_path ./dataset/ETT-small/ \
#     --data_path ETTh1.csv \
#     --save_dir save/ETTh1_RQ4_$PRED_LEN \
#     --model_id VisionTS_ETTh1_$PRED_LEN \
#     --data ETTh1 \
#     --features M \
#     --train_epochs 20 \
#     --ckpt $ckpt \
#     --vm_arch $vm_arch \
#     --lradj fixed \
#     --seq_len $context_len \
#     --periodicity $periodicity \
#     --pred_len $PRED_LEN \
#     --align_const $align_const \
#     --norm_const $norm_const \
#     --ft_type full
# done;


# for PRED_LEN in 96 192 336 720; do
#     CUDA_VISIBLE_DEVICES=5 python -u run.py \
#     --task_name long_term_forecast \
#     --learning_rate 0.0001 \
#     --batch_size 32 \
#     --is_training 1 \
#     --model VisionTS \
#     --root_path ./dataset/ETT-small/ \
#     --data_path ETTh1.csv \
#     --save_dir save/ETTh1_RQ4_$PRED_LEN \
#     --model_id VisionTS_ETTh1_$PRED_LEN \
#     --data ETTh1 \
#     --features M \
#     --train_epochs 20 \
#     --ckpt $ckpt \
#     --vm_arch $vm_arch \
#     --lradj fixed \
#     --seq_len $context_len \
#     --periodicity $periodicity \
#     --pred_len $PRED_LEN \
#     --align_const $align_const \
#     --norm_const $norm_const \
#     --ft_type full_wo_cls_mask
# done;


# for PRED_LEN in 96 192 336 720; do
#     CUDA_VISIBLE_DEVICES=5 python -u run.py \
#     --task_name long_term_forecast \
#     --learning_rate 0.0001 \
#     --batch_size 32 \
#     --is_training 1 \
#     --model VisionTS \
#     --root_path ./dataset/ETT-small/ \
#     --data_path ETTh1.csv \
#     --save_dir save/ETTh1_RQ4_$PRED_LEN \
#     --model_id VisionTS_ETTh1_$PRED_LEN \
#     --data ETTh1 \
#     --features M \
#     --train_epochs 20 \
#     --ckpt $ckpt \
#     --lradj fixed \
#     --vm_arch $vm_arch \
#     --seq_len $context_len \
#     --periodicity $periodicity \
#     --pred_len $PRED_LEN \
#     --align_const $align_const \
#     --norm_const $norm_const \
#     --ft_type mlp+ln
# done;




for PRED_LEN in 96 192 336 720; do
    CUDA_VISIBLE_DEVICES=5 python -u run.py \
    --task_name long_term_forecast \
    --learning_rate 0.0001 \
    --batch_size 32 \
    --is_training 1 \
    --model VisionTS \
    --root_path ./dataset/ETT-small/ \
    --data_path ETTh1.csv \
    --save_dir save/ETTh1_RQ4_$PRED_LEN \
    --model_id VisionTS_ETTh1_$PRED_LEN \
    --data ETTh1 \
    --features M \
    --train_epochs 0 \
    --ckpt $ckpt \
    --lradj fixed \
    --vm_arch $vm_arch \
    --seq_len $context_len \
    --periodicity $periodicity \
    --pred_len $PRED_LEN \
    --align_const $align_const \
    --norm_const $norm_const \
    --ft_type zero-shot
done;




# for PRED_LEN in 96 192 336 720; do
#     CUDA_VISIBLE_DEVICES=5 python -u run.py \
#     --task_name long_term_forecast \
#     --learning_rate 0.0001 \
#     --batch_size 32 \
#     --is_training 1 \
#     --model VisionTS \
#     --root_path ./dataset/ETT-small/ \
#     --data_path ETTh1.csv \
#     --save_dir save/ETTh1_RQ4_$PRED_LEN \
#     --model_id VisionTS_ETTh1_$PRED_LEN \
#     --data ETTh1 \
#     --features M \
#     --train_epochs 20 \
#     --lradj fixed \
#     --no_ckpt \
#     --ckpt $ckpt \
#     --vm_arch $vm_arch \
#     --seq_len $context_len \
#     --periodicity $periodicity \
#     --pred_len $PRED_LEN \
#     --align_const $align_const \
#     --norm_const $norm_const \
#     --ft_type full
# done;