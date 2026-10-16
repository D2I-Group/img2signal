export ckpt="./ckpt/"
export backbone="vit_base"
export context_len=336
export vm_dim=768
export periodicity=24
export num_features=1
export head_dim=128
export head_dropout=0.3

for PRED_LEN in 96 192 336 720; do
    CUDA_VISIBLE_DEVICES=4 python -u run.py \
    --task_name long_term_forecast \
    --learning_rate 0.0001 \
    --batch_size 64 \
    --is_training 1 \
    --model UTSRegression \
    --root_path ./dataset/traffic/ \
    --data_path traffic.csv \
    --save_dir save/traffic_$PRED_LEN \
    --model_id UTSRegression_traffic_$PRED_LEN \
    --data custom \
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