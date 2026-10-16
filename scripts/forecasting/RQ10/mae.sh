export ckpt="./ckpt/"
export vm_arch="mae_base"
export periodicity=24
export norm_const=0.4


for PRED_LEN in 96; do
    CUDA_VISIBLE_DEVICES=1 python -u run.py \
    --task_name long_term_forecast \
    --learning_rate 0.0001 \
    --batch_size 64 \
    --is_training 1 \
    --model VisionTS \
    --root_path ./dataset/ETT-small/ \
    --data_path ETTh1.csv \
    --save_dir save/ETTh1_RQ10_$PRED_LEN \
    --model_id VisionTS_ETTh1_$PRED_LEN \
    --data ETTh1 \
    --features M \
    --train_epochs 1 \
    --ckpt $ckpt \
    --lradj fixed \
    --vm_arch $vm_arch \
    --seq_len 336 \
    --periodicity $periodicity \
    --pred_len $PRED_LEN \
    --align_const 0.4 \
    --norm_const $norm_const \
    --ft_type ln
done;


export ckpt="./ckpt/"
export vm_arch="mae_base"
export context_len=336
export periodicity=144
export align_const=0.7
export norm_const=1.0
export ft_type="ln"

for PRED_LEN in 96; do
    CUDA_VISIBLE_DEVICES=1 python -u run.py \
    --task_name long_term_forecast \
    --learning_rate 0.0001 \
    --batch_size 512 \
    --is_training 1 \
    --model VisionTS \
    --root_path ./dataset/weather/ \
    --data_path weather.csv \
    --save_dir save/weather_RQ10_$PRED_LEN \
    --model_id VisionTS_weather_$PRED_LEN \
    --data custom \
    --features M \
    --train_epochs 10 \
    --ckpt $ckpt \
    --lradj fixed \
    --vm_arch $vm_arch \
    --seq_len $context_len \
    --periodicity $periodicity \
    --pred_len $PRED_LEN \
    --align_const $align_const \
    --norm_const $norm_const \
    --ft_type $ft_type
done;