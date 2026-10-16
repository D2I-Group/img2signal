export ckpt="./ckpt/"
export context_len=104
export periodicity=52
export align_const=1.0
export norm_const=0.3
export ft_type="ln"

for PRED_LEN in 24 36 48 60; do
    CUDA_VISIBLE_DEVICES=1 python -u run.py \
    --task_name long_term_forecast \
    --learning_rate 0.0001 \
    --batch_size 256 \
    --is_training 1 \
    --model MiMRegression \
    --root_path ./dataset/illness/ \
    --data_path national_illness.csv \
    --save_dir save/mim_illness_RQ7_$PRED_LEN \
    --model_id simmim_illness_$PRED_LEN \
    --data custom \
    --features M \
    --lradj fixed \
    --train_epochs 100 \
    --ckpt $ckpt \
    --seq_len $context_len \
    --periodicity $periodicity \
    --pred_len $PRED_LEN \
    --align_const $align_const \
    --norm_const $norm_const \
    --ft_type $ft_type \
    --patience 15
done;



for PRED_LEN in 24 36 48 60; do
    CUDA_VISIBLE_DEVICES=1 python -u run.py \
    --task_name long_term_forecast \
    --learning_rate 0.0001 \
    --batch_size 256 \
    --is_training 1 \
    --model MiMRegression \
    --root_path ./dataset/illness/ \
    --data_path national_illness.csv \
    --save_dir save/mim_illness_RQ7_$PRED_LEN \
    --model_id simmim_illness_$PRED_LEN \
    --data custom \
    --features M \
    --lradj fixed \
    --train_epochs 100 \
    --ckpt $ckpt \
    --seq_len $context_len \
    --periodicity $periodicity \
    --pred_len $PRED_LEN \
    --align_const $align_const \
    --norm_const $norm_const \
    --ft_type full \
    --patience 15 \
    --sep_enc_dec enc
done;



for PRED_LEN in 24 36 48 60; do
    CUDA_VISIBLE_DEVICES=1 python -u run.py \
    --task_name long_term_forecast \
    --learning_rate 0.0001 \
    --batch_size 256 \
    --is_training 1 \
    --model MiMRegression \
    --root_path ./dataset/illness/ \
    --data_path national_illness.csv \
    --save_dir save/mim_illness_RQ7_$PRED_LEN \
    --model_id simmim_illness_$PRED_LEN \
    --data custom \
    --features M \
    --lradj fixed \
    --train_epochs 100 \
    --ckpt $ckpt \
    --seq_len $context_len \
    --periodicity $periodicity \
    --pred_len $PRED_LEN \
    --align_const $align_const \
    --norm_const $norm_const \
    --ft_type full \
    --patience 15 \
    --sep_enc_dec dec
done;