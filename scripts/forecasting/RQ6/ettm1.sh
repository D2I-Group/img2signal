export ckpt="./ckpt/"
export vm_arch="mae_base"
export context_len=336
export periodicity=96
export align_const=0.4
export norm_const=0.4


for PRED_LEN in 96 192 336 720; do
    CUDA_VISIBLE_DEVICES=2 python -u run.py \
    --task_name long_term_forecast \
    --learning_rate 0.0001 \
    --batch_size 32 \
    --is_training 1 \
    --model VisionTS \
    --root_path ./dataset/ETT-small/ \
    --data_path ETTm1.csv \
    --save_dir save/ETTm1_RQ6_$PRED_LEN \
    --model_id VisionTS_ETTm1_$PRED_LEN \
    --data ETTm1 \
    --features M \
    --train_epochs 30 \
    --lradj fixed \
    --ckpt $ckpt \
    --vm_arch $vm_arch \
    --seq_len $context_len \
    --periodicity $periodicity \
    --pred_len $PRED_LEN \
    --align_const $align_const \
    --norm_const $norm_const \
    --ft_type ln
done;


for PRED_LEN in 96 192 336 720; do
    CUDA_VISIBLE_DEVICES=2 python -u run.py \
    --task_name long_term_forecast \
    --learning_rate 0.0001 \
    --batch_size 32 \
    --is_training 1 \
    --model VisionTS \
    --root_path ./dataset/ETT-small/ \
    --data_path ETTm1.csv \
    --save_dir save/ETTm1_RQ6_$PRED_LEN \
    --model_id VisionTS_ETTm1_$PRED_LEN \
    --data ETTm1 \
    --features M \
    --train_epochs 0 \
    --lradj fixed \
    --ckpt $ckpt \
    --vm_arch $vm_arch \
    --seq_len $context_len \
    --periodicity $periodicity \
    --pred_len $PRED_LEN \
    --align_const $align_const \
    --norm_const $norm_const \
    --shuffle_flag sf_all
done;


for PRED_LEN in 96 192 336 720; do
    CUDA_VISIBLE_DEVICES=2 python -u run.py \
    --task_name long_term_forecast \
    --learning_rate 0.0001 \
    --batch_size 32 \
    --is_training 1 \
    --model VisionTS \
    --root_path ./dataset/ETT-small/ \
    --data_path ETTm1.csv \
    --save_dir save/ETTm1_RQ6_$PRED_LEN \
    --model_id VisionTS_ETTm1_$PRED_LEN \
    --data ETTm1 \
    --features M \
    --train_epochs 0 \
    --lradj fixed \
    --ckpt $ckpt \
    --vm_arch $vm_arch \
    --seq_len $context_len \
    --periodicity $periodicity \
    --pred_len $PRED_LEN \
    --align_const $align_const \
    --norm_const $norm_const \
    --shuffle_flag sf_half
done;


for PRED_LEN in 96 192 336 720; do
    CUDA_VISIBLE_DEVICES=2 python -u run.py \
    --task_name long_term_forecast \
    --learning_rate 0.0001 \
    --batch_size 32 \
    --is_training 1 \
    --model VisionTS \
    --root_path ./dataset/ETT-small/ \
    --data_path ETTm1.csv \
    --save_dir save/ETTm1_RQ6_$PRED_LEN \
    --model_id VisionTS_ETTm1_$PRED_LEN \
    --data ETTm1 \
    --features M \
    --train_epochs 0 \
    --lradj fixed \
    --ckpt $ckpt \
    --vm_arch $vm_arch \
    --seq_len $context_len \
    --periodicity $periodicity \
    --pred_len $PRED_LEN \
    --align_const $align_const \
    --norm_const $norm_const \
    --shuffle_flag ex_half
done;


for PRED_LEN in 96 192 336 720; do
    CUDA_VISIBLE_DEVICES=2 python -u run.py \
    --task_name long_term_forecast \
    --learning_rate 0.0001 \
    --batch_size 32 \
    --is_training 1 \
    --model VisionTS \
    --root_path ./dataset/ETT-small/ \
    --data_path ETTm1.csv \
    --save_dir save/ETTm1_RQ6_$PRED_LEN \
    --model_id VisionTS_ETTm1_$PRED_LEN \
    --data ETTm1 \
    --features M \
    --train_epochs 0 \
    --lradj fixed \
    --ckpt $ckpt \
    --vm_arch $vm_arch \
    --seq_len $context_len \
    --periodicity $periodicity \
    --pred_len $PRED_LEN \
    --align_const $align_const \
    --norm_const $norm_const \
    --shuffle_flag mask
done;


export context_len=336
export periodicity=96
export num_features=7
export head_dim=128
export head_dropout=0.3
export interpolation="bilinear"


for PRED_LEN in 96 192 336 720; do
    CUDA_VISIBLE_DEVICES=2 python -u run.py \
    --task_name long_term_forecast \
    --learning_rate 0.0001 \
    --batch_size 32 \
    --is_training 1 \
    --model Linear_RQ5 \
    --root_path ./dataset/ETT-small/ \
    --data_path ETTm1.csv \
    --save_dir save/ETTm1_Linear_RQ6_$PRED_LEN \
    --model_id Linear_RQ6_ETTm1_$PRED_LEN \
    --data ETTm1 \
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
    CUDA_VISIBLE_DEVICES=2 python -u run.py \
    --task_name long_term_forecast \
    --learning_rate 0.0001 \
    --batch_size 32 \
    --is_training 1 \
    --model Linear_RQ5 \
    --root_path ./dataset/ETT-small/ \
    --data_path ETTm1.csv \
    --save_dir save/ETTm1_Linear_RQ6_$PRED_LEN \
    --model_id Linear_RQ6_ETTm1_$PRED_LEN \
    --data ETTm1 \
    --features M \
    --train_epochs 0 \
    --context_len $context_len \
    --seq_len $context_len \
    --lradj fixed \
    --pred_len $PRED_LEN \
    --periodicity $periodicity \
    --num_features $num_features \
    --head_dim $head_dim \
    --head_dropout $head_dropout \
    --shuffle_flag sf_all
done;


for PRED_LEN in 96 192 336 720; do
    CUDA_VISIBLE_DEVICES=2 python -u run.py \
    --task_name long_term_forecast \
    --learning_rate 0.0001 \
    --batch_size 32 \
    --is_training 1 \
    --model Linear_RQ5 \
    --root_path ./dataset/ETT-small/ \
    --data_path ETTm1.csv \
    --save_dir save/ETTm1_Linear_RQ6_$PRED_LEN \
    --model_id Linear_RQ6_ETTm1_$PRED_LEN \
    --data ETTm1 \
    --features M \
    --train_epochs 0 \
    --context_len $context_len \
    --seq_len $context_len \
    --lradj fixed \
    --pred_len $PRED_LEN \
    --periodicity $periodicity \
    --num_features $num_features \
    --head_dim $head_dim \
    --head_dropout $head_dropout \
    --shuffle_flag sf_half
done;



for PRED_LEN in 96 192 336 720; do
    CUDA_VISIBLE_DEVICES=2 python -u run.py \
    --task_name long_term_forecast \
    --learning_rate 0.0001 \
    --batch_size 32 \
    --is_training 1 \
    --model Linear_RQ5 \
    --root_path ./dataset/ETT-small/ \
    --data_path ETTm1.csv \
    --save_dir save/ETTm1_Linear_RQ6_$PRED_LEN \
    --model_id Linear_RQ6_ETTm1_$PRED_LEN \
    --data ETTm1 \
    --features M \
    --train_epochs 0 \
    --context_len $context_len \
    --seq_len $context_len \
    --lradj fixed \
    --pred_len $PRED_LEN \
    --periodicity $periodicity \
    --num_features $num_features \
    --head_dim $head_dim \
    --head_dropout $head_dropout \
    --shuffle_flag ex_half
done;


for PRED_LEN in 96 192 336 720; do
    CUDA_VISIBLE_DEVICES=2 python -u run.py \
    --task_name long_term_forecast \
    --learning_rate 0.0001 \
    --batch_size 32 \
    --is_training 1 \
    --model Linear_RQ5 \
    --root_path ./dataset/ETT-small/ \
    --data_path ETTm1.csv \
    --save_dir save/ETTm1_Linear_RQ6_$PRED_LEN \
    --model_id Linear_RQ6_ETTm1_$PRED_LEN \
    --data ETTm1 \
    --features M \
    --train_epochs 0 \
    --context_len $context_len \
    --seq_len $context_len \
    --lradj fixed \
    --pred_len $PRED_LEN \
    --periodicity $periodicity \
    --num_features $num_features \
    --head_dim $head_dim \
    --head_dropout $head_dropout \
    --shuffle_flag mask
done;



for PRED_LEN in 96 192 336 720; do
    CUDA_VISIBLE_DEVICES=2 python -u run.py \
    --task_name long_term_forecast \
    --learning_rate 0.0001 \
    --batch_size 32 \
    --is_training 1 \
    --model Attn_RQ5 \
    --root_path ./dataset/ETT-small/ \
    --data_path ETTm1.csv \
    --save_dir save/ETTm1_RQ6_Attn_$PRED_LEN \
    --model_id Attn_RQ6_ETTm1_$PRED_LEN \
    --data ETTm1 \
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
    CUDA_VISIBLE_DEVICES=2 python -u run.py \
    --task_name long_term_forecast \
    --learning_rate 0.0001 \
    --batch_size 32 \
    --is_training 1 \
    --model Attn_RQ5 \
    --root_path ./dataset/ETT-small/ \
    --data_path ETTm1.csv \
    --save_dir save/ETTm1_RQ6_Attn_$PRED_LEN \
    --model_id Attn_RQ6_ETTm1_$PRED_LEN \
    --data ETTm1 \
    --features M \
    --train_epochs 0 \
    --context_len $context_len \
    --seq_len $context_len \
    --lradj fixed \
    --pred_len $PRED_LEN \
    --periodicity $periodicity \
    --num_features $num_features \
    --head_dim $head_dim \
    --head_dropout $head_dropout \
    --shuffle_flag sf_all
done;


for PRED_LEN in 96 192 336 720; do
    CUDA_VISIBLE_DEVICES=2 python -u run.py \
    --task_name long_term_forecast \
    --learning_rate 0.0001 \
    --batch_size 32 \
    --is_training 1 \
    --model Attn_RQ5 \
    --root_path ./dataset/ETT-small/ \
    --data_path ETTm1.csv \
    --save_dir save/ETTm1_RQ6_Attn_$PRED_LEN \
    --model_id Attn_RQ6_ETTm1_$PRED_LEN \
    --data ETTm1 \
    --features M \
    --train_epochs 0 \
    --context_len $context_len \
    --seq_len $context_len \
    --lradj fixed \
    --pred_len $PRED_LEN \
    --periodicity $periodicity \
    --num_features $num_features \
    --head_dim $head_dim \
    --head_dropout $head_dropout \
    --shuffle_flag sf_half
done;




for PRED_LEN in 96 192 336 720; do
    CUDA_VISIBLE_DEVICES=2 python -u run.py \
    --task_name long_term_forecast \
    --learning_rate 0.0001 \
    --batch_size 32 \
    --is_training 1 \
    --model Attn_RQ5 \
    --root_path ./dataset/ETT-small/ \
    --data_path ETTm1.csv \
    --save_dir save/ETTm1_RQ6_Attn_$PRED_LEN \
    --model_id Attn_RQ6_ETTm1_$PRED_LEN \
    --data ETTm1 \
    --features M \
    --train_epochs 0 \
    --context_len $context_len \
    --seq_len $context_len \
    --lradj fixed \
    --pred_len $PRED_LEN \
    --periodicity $periodicity \
    --num_features $num_features \
    --head_dim $head_dim \
    --head_dropout $head_dropout \
    --shuffle_flag ex_half
done;


for PRED_LEN in 96 192 336 720; do
    CUDA_VISIBLE_DEVICES=2 python -u run.py \
    --task_name long_term_forecast \
    --learning_rate 0.0001 \
    --batch_size 32 \
    --is_training 1 \
    --model Attn_RQ5 \
    --root_path ./dataset/ETT-small/ \
    --data_path ETTm1.csv \
    --save_dir save/ETTm1_RQ6_Attn_$PRED_LEN \
    --model_id Attn_RQ6_ETTm1_$PRED_LEN \
    --data ETTm1 \
    --features M \
    --train_epochs 0 \
    --context_len $context_len \
    --seq_len $context_len \
    --lradj fixed \
    --pred_len $PRED_LEN \
    --periodicity $periodicity \
    --num_features $num_features \
    --head_dim $head_dim \
    --head_dropout $head_dropout \
    --shuffle_flag mask
done;


