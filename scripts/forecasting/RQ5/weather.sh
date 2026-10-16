export context_len=336
export periodicity=144
export num_features=1
export head_dim=128
export head_dropout=0.3
export interpolation="bilinear"


for PRED_LEN in 96 192 336 720; do
    CUDA_VISIBLE_DEVICES=7 python -u run.py \
    --task_name long_term_forecast \
    --learning_rate 0.0001 \
    --batch_size 1024 \
    --is_training 1 \
    --model Linear_RQ5 \
    --root_path ./dataset/weather/ \
    --data_path weather.csv \
    --save_dir save/weather_RQ5_$PRED_LEN \
    --model_id Linear_RQ5_weather_$PRED_LEN \
    --data custom \
    --features M \
    --train_epochs 20 \
    --context_len $context_len \
    --seq_len $context_len \
    --lradj fixed \
    --pred_len $PRED_LEN \
    --periodicity $periodicity \
    --num_features $num_features \
    --head_dim $head_dim \
    --head_dropout $head_dropout
done;


for PRED_LEN in 96 192 336 720; do
    CUDA_VISIBLE_DEVICES=7 python -u run.py \
    --task_name long_term_forecast \
    --learning_rate 0.0001 \
    --batch_size 1024 \
    --is_training 1 \
    --model Linear_RQ5 \
    --root_path ./dataset/weather/ \
    --data_path weather.csv \
    --save_dir save/weather_RQ5_$PRED_LEN \
    --model_id Linear_RQ5_weather_$PRED_LEN \
    --data custom \
    --features M \
    --train_epochs 20 \
    --context_len $context_len \
    --seq_len $context_len \
    --lradj fixed \
    --pred_len $PRED_LEN \
    --periodicity $periodicity \
    --num_features $num_features \
    --head_dim $head_dim \
    --head_dropout $head_dropout
done;