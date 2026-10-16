export ckpt="./ckpt/"
export vm_arch="mae_base"
export context_len=336
export align_const=0.7
export norm_const=1.0


for PRED_LEN in 96 192 336 720; do
    CUDA_VISIBLE_DEVICES=2 python -u run.py \
    --task_name long_term_forecast \
    --learning_rate 0.0001 \
    --batch_size 256 \
    --is_training 1 \
    --model VisionTS \
    --root_path ./dataset/weather/ \
    --data_path weather.csv \
    --save_dir save/weather_RQ8_120_192_$PRED_LEN \
    --model_id VisionTS_weather_$PRED_LEN \
    --data custom \
    --features M \
    --train_epochs 10 \
    --ckpt $ckpt \
    --lradj fixed \
    --vm_arch $vm_arch \
    --seq_len $context_len \
    --periodicity 120 \
    --pred_len $PRED_LEN \
    --align_const $align_const \
    --norm_const $norm_const \
    --ft_type ln
done;

for PRED_LEN in 96 192 336 720; do
    CUDA_VISIBLE_DEVICES=2 python -u run.py \
    --task_name long_term_forecast \
    --learning_rate 0.0001 \
    --batch_size 256 \
    --is_training 1 \
    --model VisionTS \
    --root_path ./dataset/weather/ \
    --data_path weather.csv \
    --save_dir save/weather_RQ8_120_192_$PRED_LEN \
    --model_id VisionTS_weather_$PRED_LEN \
    --data custom \
    --features M \
    --train_epochs 10 \
    --ckpt $ckpt \
    --lradj fixed \
    --vm_arch $vm_arch \
    --seq_len $context_len \
    --periodicity 168 \
    --pred_len $PRED_LEN \
    --align_const $align_const \
    --norm_const $norm_const \
    --ft_type ln
done;



for PRED_LEN in 96 192 336 720; do
    CUDA_VISIBLE_DEVICES=2 python -u run.py \
    --task_name long_term_forecast \
    --learning_rate 0.0001 \
    --batch_size 256 \
    --is_training 1 \
    --model VisionTS \
    --root_path ./dataset/weather/ \
    --data_path weather.csv \
    --save_dir save/weather_RQ8_120_192_$PRED_LEN \
    --model_id VisionTS_weather_$PRED_LEN \
    --data custom \
    --features M \
    --train_epochs 10 \
    --ckpt $ckpt \
    --lradj fixed \
    --vm_arch $vm_arch \
    --seq_len $context_len \
    --periodicity 192 \
    --pred_len $PRED_LEN \
    --align_const $align_const \
    --norm_const $norm_const \
    --ft_type ln
done;
