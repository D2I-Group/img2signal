export ckpt="./ckpt/"
export vm_arch="mae_base"
export context_len=336
export periodicity=96
export align_const=0.4
export norm_const=0.4
export ft_type="ln"

for PRED_LEN in 96 192 336 720; do
    CUDA_VISIBLE_DEVICES=7 python -u run.py \
    --task_name long_term_forecast \
    --learning_rate 0.0001 \
    --batch_size 32 \
    --is_training 1 \
    --model VisionTS \
    --root_path ./dataset/ETT-small/ \
    --data_path ETTm1.csv \
    --save_dir save/ETTm1_$PRED_LEN \
    --model_id VisionTS_ETTm1_$PRED_LEN \
    --data ETTm1 \
    --features M \
    --train_epochs 30 \
    --ckpt $ckpt \
    --vm_arch $vm_arch \
    --seq_len $context_len \
    --periodicity $periodicity \
    --pred_len $PRED_LEN \
    --align_const $align_const \
    --norm_const $norm_const \
    --ft_type $ft_type
done;