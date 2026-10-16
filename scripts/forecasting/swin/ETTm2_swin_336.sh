export ckpt="./ckpt/"
export backbone="swin_tiny"
export context_len=336
export vm_dim=768
export periodicity=96
export num_features=7
export head_dim=128
export head_dropout=0.3

for PRED_LEN in 96 192 336 720; do
    CUDA_VISIBLE_DEVICES=6 python -u run.py \
    --task_name long_term_forecast \
    --learning_rate 0.0001 \
    --batch_size 32 \
    --is_training 1 \
    --model UTSRegression \
    --root_path ./dataset/ETT-small/ \
    --data_path ETTm2.csv \
    --save_dir save/ETTm2_$PRED_LEN \
    --model_id UTSRegression_ETTm2_$PRED_LEN \
    --data ETTm2 \
    --features M \
    --train_epochs 20 \
    --ckpt $ckpt \
    --context_len $context_len \
    --seq_len $context_len \
    --lradj fixed \
    --pred_len $PRED_LEN \
    --vm_dim $vm_dim \
    --periodicity $periodicity \
    --num_features $num_features \
    --head_dim $head_dim \
    --head_dropout $head_dropout \
    --backbone $backbone
done;