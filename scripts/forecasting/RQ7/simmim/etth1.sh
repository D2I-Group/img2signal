export ckpt="./ckpt/"
export context_len=336
export periodicity=24
export align_const=1.0
export norm_const=0.3


for PRED_LEN in 96 192 336 720; do
    CUDA_VISIBLE_DEVICES=1 python -u run.py \
    --task_name long_term_forecast \
    --learning_rate 0.0001 \
    --batch_size 32 \
    --is_training 1 \
    --model MiMRegression \
    --root_path ./dataset/ETT-small/ \
    --data_path ETTh1.csv \
    --save_dir save/mim_ETTh1_RQ7_$PRED_LEN \
    --model_id simmim_ETTh1_RQ7_$PRED_LEN \
    --data ETTh1 \
    --features M \
    --train_epochs 5 \
    --ckpt $ckpt \
    --seq_len $context_len \
    --periodicity $periodicity \
    --pred_len $PRED_LEN \
    --align_const $align_const \
    --norm_const $norm_const \
    --ft_type ln \
    --patience 3
done;


for PRED_LEN in 96 192 336 720; do
    CUDA_VISIBLE_DEVICES=1 python -u run.py \
    --task_name long_term_forecast \
    --learning_rate 0.0001 \
    --batch_size 32 \
    --is_training 1 \
    --model MiMRegression \
    --root_path ./dataset/ETT-small/ \
    --data_path ETTh1.csv \
    --save_dir save/mim_ETTh1_RQ7_$PRED_LEN \
    --model_id simmim_ETTh1_RQ7_$PRED_LEN \
    --data ETTh1 \
    --features M \
    --train_epochs 5 \
    --ckpt $ckpt \
    --seq_len $context_len \
    --periodicity $periodicity \
    --pred_len $PRED_LEN \
    --align_const $align_const \
    --norm_const $norm_const \
    --ft_type full \
    --patience 3 \
    --sep_enc_dec enc
done;



for PRED_LEN in 96 192 336 720; do
    CUDA_VISIBLE_DEVICES=1 python -u run.py \
    --task_name long_term_forecast \
    --learning_rate 0.0001 \
    --batch_size 32 \
    --is_training 1 \
    --model MiMRegression \
    --root_path ./dataset/ETT-small/ \
    --data_path ETTh1.csv \
    --save_dir save/mim_ETTh1_RQ7_$PRED_LEN \
    --model_id simmim_ETTh1_RQ7_$PRED_LEN \
    --data ETTh1 \
    --features M \
    --train_epochs 5 \
    --ckpt $ckpt \
    --seq_len $context_len \
    --periodicity $periodicity \
    --pred_len $PRED_LEN \
    --align_const $align_const \
    --norm_const $norm_const \
    --ft_type full \
    --patience 3 \
    --sep_enc_dec dec
done;