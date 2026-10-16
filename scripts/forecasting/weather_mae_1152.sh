export ckpt="./ckpt/"
export vm_arch="mae_base"
export context_len=576
export periodicity=144
export align_const=0.7
export norm_const=1.0
export ft_type="ln"

for PRED_LEN in 96 192 336 720; do
    CUDA_VISIBLE_DEVICES=5 python -u run.py \
    --task_name long_term_forecast \
    --learning_rate 0.0001 \
    --batch_size 512 \
    --is_training 1 \
    --model VisionTS \
    --root_path ./dataset/weather/ \
    --data_path weather.csv \
    --save_dir save/weather_$PRED_LEN \
    --model_id VisionTS_weather_$PRED_LEN \
    --data custom \
    --features M \
    --train_epochs 3 \
    --ckpt $ckpt \
    --vm_arch $vm_arch \
    --seq_len $context_len \
    --periodicity $periodicity \
    --pred_len $PRED_LEN \
    --align_const $align_const \
    --norm_const $norm_const \
    --ft_type $ft_type
done;