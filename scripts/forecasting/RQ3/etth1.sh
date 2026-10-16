export ckpt="./ckpt/"
export vm_arch="mae_base"
export context_len=336
export ft_type="ln"
export vm_dim=768
export num_features=7
export head_dim=128
export head_dropout=0.3

for PRED_LEN in 96 192 336 720; do
    CUDA_VISIBLE_DEVICES=1 python -u run.py \
    --task_name long_term_forecast \
    --learning_rate 0.00001 \
    --batch_size 32 \
    --is_training 1 \
    --model MAE_RQ3 \
    --time2img GAF \
    --root_path ./dataset/ETT-small/ \
    --data_path ETTh1.csv \
    --save_dir save/ETTh1_$PRED_LEN \
    --model_id MAE_RQ3_ETTh1_$PRED_LEN \
    --data ETTh1 \
    --features M \
    --lradj fixed \
    --train_epochs 20 \
    --ckpt $ckpt \
    --vm_arch $vm_arch \
    --seq_len $context_len \
    --pred_len $PRED_LEN \
    --ft_type $ft_type \
    --num_features $num_features \
    --head_dim $head_dim \
    --head_dropout $head_dropout 
done;


for PRED_LEN in 96 192 336 720; do
    CUDA_VISIBLE_DEVICES=1 python -u run.py \
    --task_name long_term_forecast \
    --learning_rate 0.00001 \
    --batch_size 32 \
    --is_training 1 \
    --model MAE_RQ3 \
    --time2img RP \
    --root_path ./dataset/ETT-small/ \
    --data_path ETTh1.csv \
    --save_dir save/ETTh1_$PRED_LEN \
    --model_id MAE_RQ3_ETTh1_$PRED_LEN \
    --data ETTh1 \
    --features M \
    --lradj fixed \
    --train_epochs 20 \
    --ckpt $ckpt \
    --vm_arch $vm_arch \
    --seq_len $context_len \
    --pred_len $PRED_LEN \
    --ft_type $ft_type \
    --num_features $num_features \
    --head_dim $head_dim \
    --head_dropout $head_dropout 
done;

for PRED_LEN in 96 192 336 720; do
    CUDA_VISIBLE_DEVICES=1 python -u run.py \
    --task_name long_term_forecast \
    --learning_rate 0.00001 \
    --batch_size 32 \
    --is_training 1 \
    --model MAE_RQ3 \
    --time2img STFT \
    --root_path ./dataset/ETT-small/ \
    --data_path ETTh1.csv \
    --save_dir save/ETTh1_$PRED_LEN \
    --model_id MAE_RQ3_ETTh1_$PRED_LEN \
    --data ETTh1 \
    --features M \
    --lradj fixed \
    --train_epochs 20 \
    --ckpt $ckpt \
    --vm_arch $vm_arch \
    --seq_len $context_len \
    --pred_len $PRED_LEN \
    --ft_type $ft_type \
    --num_features $num_features \
    --head_dim $head_dim \
    --head_dropout $head_dropout 
done;

for PRED_LEN in 96 192 336 720; do
    CUDA_VISIBLE_DEVICES=1 python -u run.py \
    --task_name long_term_forecast \
    --learning_rate 0.00001 \
    --batch_size 32 \
    --is_training 1 \
    --model MAE_RQ3 \
    --time2img Wavelet \
    --root_path ./dataset/ETT-small/ \
    --data_path ETTh1.csv \
    --save_dir save/ETTh1_$PRED_LEN \
    --model_id MAE_RQ3_ETTh1_$PRED_LEN \
    --data ETTh1 \
    --features M \
    --lradj fixed \
    --train_epochs 20 \
    --ckpt $ckpt \
    --vm_arch $vm_arch \
    --seq_len $context_len \
    --pred_len $PRED_LEN \
    --ft_type $ft_type \
    --num_features $num_features \
    --head_dim $head_dim \
    --head_dropout $head_dropout 
done;


for PRED_LEN in 96 192 336 720; do
    CUDA_VISIBLE_DEVICES=1 python -u run.py \
    --task_name long_term_forecast \
    --learning_rate 0.00001 \
    --batch_size 32 \
    --is_training 1 \
    --model MAE_RQ3 \
    --time2img Filterbank \
    --root_path ./dataset/ETT-small/ \
    --data_path ETTh1.csv \
    --save_dir save/ETTh1_$PRED_LEN \
    --model_id MAE_RQ3_ETTh1_$PRED_LEN \
    --data ETTh1 \
    --features M \
    --lradj fixed \
    --train_epochs 20 \
    --ckpt $ckpt \
    --vm_arch $vm_arch \
    --seq_len $context_len \
    --pred_len $PRED_LEN \
    --ft_type $ft_type \
    --num_features $num_features \
    --head_dim $head_dim \
    --head_dropout $head_dropout 
done;


export ckpt="./ckpt/"
export vm_arch="mae_base"
export context_len=336
export align_const=0.4
export norm_const=1
export ft_type="ln"
export num_features=7

for PRED_LEN in 96 192 336 720; do
    CUDA_VISIBLE_DEVICES=1 python -u run.py \
    --task_name long_term_forecast \
    --learning_rate 0.0001 \
    --batch_size 32 \
    --is_training 1 \
    --model MVH_RQ3 \
    --root_path ./dataset/ETT-small/ \
    --data_path ETTh1.csv \
    --save_dir save/ETTh1_$PRED_LEN \
    --model_id MVH_RQ3_ETTh1_$PRED_LEN \
    --data ETTh1 \
    --features M \
    --lradj fixed \
    --train_epochs 20 \
    --ckpt $ckpt \
    --vm_arch $vm_arch \
    --seq_len $context_len \
    --pred_len $PRED_LEN \
    --ft_type $ft_type \
    --num_features $num_features \
    --align_const $align_const \
    --norm_const $norm_const 
done;