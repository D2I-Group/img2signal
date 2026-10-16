export ckpt="./ckpt/"
export context_len=336
export periodicity=96
export align_const=1.0
export norm_const=0.3
export ft_type="ln"

for PRED_LEN in 96 192 336 720; do
    CUDA_VISIBLE_DEVICES=2 python -u run.py \
    --task_name long_term_forecast \
    --learning_rate 0.0001 \
    --batch_size 32 \
    --is_training 1 \
    --model MiMRegression \
    --root_path ./dataset/ETT-small/ \
    --data_path ETTm2.csv \
    --save_dir save/ETTm2_$PRED_LEN \
    --model_id simmim_ETTm2_$PRED_LEN \
    --data ETTm2 \
    --features M \
    --train_epochs 5 \
    --ckpt $ckpt \
    --seq_len $context_len \
    --periodicity $periodicity \
    --pred_len $PRED_LEN \
    --align_const $align_const \
    --norm_const $norm_const \
    --ft_type $ft_type \
    --patience 3
done;