export ckpt="./ckpt/"
export vm_arch="mae_base"
export periodicity=144
export align_const=0.7
export norm_const=1.0


for PRED_LEN in 96 192 336 720; do
    CUDA_VISIBLE_DEVICES=7 python -u run.py \
    --task_name long_term_forecast \
    --learning_rate 0.0001 \
    --batch_size 256 \
    --is_training 1 \
    --model VisionTS \
    --root_path ./dataset/weather/ \
    --data_path weather.csv \
    --save_dir save/weather_RQ9_$PRED_LEN \
    --model_id VisionTS_weather_$PRED_LEN \
    --data custom \
    --features M \
    --train_epochs 10 \
    --ckpt $ckpt \
    --lradj fixed \
    --vm_arch $vm_arch \
    --seq_len 672 \
    --periodicity $periodicity \
    --pred_len $PRED_LEN \
    --align_const 0.5 \
    --norm_const $norm_const \
    --ft_type ln
done;

for PRED_LEN in 96 192 336 720; do
    CUDA_VISIBLE_DEVICES=7 python -u run.py \
    --task_name long_term_forecast \
    --learning_rate 0.0001 \
    --batch_size 256 \
    --is_training 1 \
    --model VisionTS \
    --root_path ./dataset/weather/ \
    --data_path weather.csv \
    --save_dir save/weather_RQ9_$PRED_LEN \
    --model_id VisionTS_weather_$PRED_LEN \
    --data custom \
    --features M \
    --train_epochs 10 \
    --ckpt $ckpt \
    --lradj fixed \
    --vm_arch $vm_arch \
    --seq_len 1728 \
    --periodicity $periodicity \
    --pred_len $PRED_LEN \
    --align_const 0.4 \
    --norm_const $norm_const \
    --ft_type ln
done;


