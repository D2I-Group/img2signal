export ckpt="./ckpt/"
export vm_arch="mae_base"
export context_len=336
export align_const=0.4
export norm_const=0.4

for PRED_LEN in 96 192 336 720; do
    CUDA_VISIBLE_DEVICES=6 python -u run.py \
    --task_name long_term_forecast \
    --learning_rate 0.0001 \
    --batch_size 32 \
    --is_training 1 \
    --model VisionTS \
    --root_path ./dataset/ETT-small/ \
    --data_path ETTm2.csv \
    --save_dir save/ETTm2_RQ8_$PRED_LEN \
    --model_id VisionTS_ETTm2_$PRED_LEN \
    --data ETTm2 \
    --features M \
    --train_epochs 30 \
    --lradj fixed \
    --ckpt $ckpt \
    --vm_arch $vm_arch \
    --seq_len $context_len \
    --periodicity 16 \
    --pred_len $PRED_LEN \
    --align_const $align_const \
    --norm_const $norm_const \
    --ft_type ln
done;


for PRED_LEN in 96 192 336 720; do
    CUDA_VISIBLE_DEVICES=6 python -u run.py \
    --task_name long_term_forecast \
    --learning_rate 0.0001 \
    --batch_size 32 \
    --is_training 1 \
    --model VisionTS \
    --root_path ./dataset/ETT-small/ \
    --data_path ETTm2.csv \
    --save_dir save/ETTm2_RQ8_$PRED_LEN \
    --model_id VisionTS_ETTm2_$PRED_LEN \
    --data ETTm2 \
    --features M \
    --train_epochs 30 \
    --lradj fixed \
    --ckpt $ckpt \
    --vm_arch $vm_arch \
    --seq_len $context_len \
    --periodicity 32 \
    --pred_len $PRED_LEN \
    --align_const $align_const \
    --norm_const $norm_const \
    --ft_type ln
done;


for PRED_LEN in 96 192 336 720; do
    CUDA_VISIBLE_DEVICES=6 python -u run.py \
    --task_name long_term_forecast \
    --learning_rate 0.0001 \
    --batch_size 32 \
    --is_training 1 \
    --model VisionTS \
    --root_path ./dataset/ETT-small/ \
    --data_path ETTm2.csv \
    --save_dir save/ETTm2_RQ8_$PRED_LEN \
    --model_id VisionTS_ETTm2_$PRED_LEN \
    --data ETTm2 \
    --features M \
    --train_epochs 30 \
    --lradj fixed \
    --ckpt $ckpt \
    --vm_arch $vm_arch \
    --seq_len $context_len \
    --periodicity 48 \
    --pred_len $PRED_LEN \
    --align_const $align_const \
    --norm_const $norm_const \
    --ft_type ln
done;

for PRED_LEN in 96 192 336 720; do
    CUDA_VISIBLE_DEVICES=6 python -u run.py \
    --task_name long_term_forecast \
    --learning_rate 0.0001 \
    --batch_size 32 \
    --is_training 1 \
    --model VisionTS \
    --root_path ./dataset/ETT-small/ \
    --data_path ETTm2.csv \
    --save_dir save/ETTm2_RQ8_$PRED_LEN \
    --model_id VisionTS_ETTm2_$PRED_LEN \
    --data ETTm2 \
    --features M \
    --train_epochs 30 \
    --lradj fixed \
    --ckpt $ckpt \
    --vm_arch $vm_arch \
    --seq_len $context_len \
    --periodicity 64 \
    --pred_len $PRED_LEN \
    --align_const $align_const \
    --norm_const $norm_const \
    --ft_type ln
done;

for PRED_LEN in 96 192 336 720; do
    CUDA_VISIBLE_DEVICES=6 python -u run.py \
    --task_name long_term_forecast \
    --learning_rate 0.0001 \
    --batch_size 32 \
    --is_training 1 \
    --model VisionTS \
    --root_path ./dataset/ETT-small/ \
    --data_path ETTm2.csv \
    --save_dir save/ETTm2_RQ8_$PRED_LEN \
    --model_id VisionTS_ETTm2_$PRED_LEN \
    --data ETTm2 \
    --features M \
    --train_epochs 30 \
    --lradj fixed \
    --ckpt $ckpt \
    --vm_arch $vm_arch \
    --seq_len $context_len \
    --periodicity 80 \
    --pred_len $PRED_LEN \
    --align_const $align_const \
    --norm_const $norm_const \
    --ft_type ln
done;

for PRED_LEN in 96 192 336 720; do
    CUDA_VISIBLE_DEVICES=6 python -u run.py \
    --task_name long_term_forecast \
    --learning_rate 0.0001 \
    --batch_size 32 \
    --is_training 1 \
    --model VisionTS \
    --root_path ./dataset/ETT-small/ \
    --data_path ETTm2.csv \
    --save_dir save/ETTm2_RQ8_$PRED_LEN \
    --model_id VisionTS_ETTm2_$PRED_LEN \
    --data ETTm2 \
    --features M \
    --train_epochs 30 \
    --lradj fixed \
    --ckpt $ckpt \
    --vm_arch $vm_arch \
    --seq_len $context_len \
    --periodicity 80 \
    --pred_len $PRED_LEN \
    --align_const $align_const \
    --norm_const $norm_const \
    --ft_type ln
done;

for PRED_LEN in 96 192 336 720; do
    CUDA_VISIBLE_DEVICES=6 python -u run.py \
    --task_name long_term_forecast \
    --learning_rate 0.0001 \
    --batch_size 32 \
    --is_training 1 \
    --model VisionTS \
    --root_path ./dataset/ETT-small/ \
    --data_path ETTm2.csv \
    --save_dir save/ETTm2_RQ8_$PRED_LEN \
    --model_id VisionTS_ETTm2_$PRED_LEN \
    --data ETTm2 \
    --features M \
    --train_epochs 30 \
    --lradj fixed \
    --ckpt $ckpt \
    --vm_arch $vm_arch \
    --seq_len $context_len \
    --periodicity 112 \
    --pred_len $PRED_LEN \
    --align_const $align_const \
    --norm_const $norm_const \
    --ft_type ln
done;


for PRED_LEN in 96 192 336 720; do
    CUDA_VISIBLE_DEVICES=6 python -u run.py \
    --task_name long_term_forecast \
    --learning_rate 0.0001 \
    --batch_size 32 \
    --is_training 1 \
    --model VisionTS \
    --root_path ./dataset/ETT-small/ \
    --data_path ETTm2.csv \
    --save_dir save/ETTm2_RQ8_$PRED_LEN \
    --model_id VisionTS_ETTm2_$PRED_LEN \
    --data ETTm2 \
    --features M \
    --train_epochs 30 \
    --lradj fixed \
    --ckpt $ckpt \
    --vm_arch $vm_arch \
    --seq_len $context_len \
    --periodicity 128 \
    --pred_len $PRED_LEN \
    --align_const $align_const \
    --norm_const $norm_const \
    --ft_type ln
done;


for PRED_LEN in 96 192 336 720; do
    CUDA_VISIBLE_DEVICES=6 python -u run.py \
    --task_name long_term_forecast \
    --learning_rate 0.0001 \
    --batch_size 32 \
    --is_training 1 \
    --model VisionTS \
    --root_path ./dataset/ETT-small/ \
    --data_path ETTm2.csv \
    --save_dir save/ETTm2_RQ8_$PRED_LEN \
    --model_id VisionTS_ETTm2_$PRED_LEN \
    --data ETTm2 \
    --features M \
    --train_epochs 30 \
    --lradj fixed \
    --ckpt $ckpt \
    --vm_arch $vm_arch \
    --seq_len $context_len \
    --periodicity 144 \
    --pred_len $PRED_LEN \
    --align_const $align_const \
    --norm_const $norm_const \
    --ft_type ln
done;



for PRED_LEN in 96 192 336 720; do
    CUDA_VISIBLE_DEVICES=6 python -u run.py \
    --task_name long_term_forecast \
    --learning_rate 0.0001 \
    --batch_size 32 \
    --is_training 1 \
    --model VisionTS \
    --root_path ./dataset/ETT-small/ \
    --data_path ETTm2.csv \
    --save_dir save/ETTm2_RQ8_$PRED_LEN \
    --model_id VisionTS_ETTm2_$PRED_LEN \
    --data ETTm2 \
    --features M \
    --train_epochs 30 \
    --lradj fixed \
    --ckpt $ckpt \
    --vm_arch $vm_arch \
    --seq_len $context_len \
    --periodicity 160 \
    --pred_len $PRED_LEN \
    --align_const $align_const \
    --norm_const $norm_const \
    --ft_type ln
done;

for PRED_LEN in 96 192 336 720; do
    CUDA_VISIBLE_DEVICES=6 python -u run.py \
    --task_name long_term_forecast \
    --learning_rate 0.0001 \
    --batch_size 32 \
    --is_training 1 \
    --model VisionTS \
    --root_path ./dataset/ETT-small/ \
    --data_path ETTm2.csv \
    --save_dir save/ETTm2_RQ8_$PRED_LEN \
    --model_id VisionTS_ETTm2_$PRED_LEN \
    --data ETTm2 \
    --features M \
    --train_epochs 30 \
    --lradj fixed \
    --ckpt $ckpt \
    --vm_arch $vm_arch \
    --seq_len $context_len \
    --periodicity 176 \
    --pred_len $PRED_LEN \
    --align_const $align_const \
    --norm_const $norm_const \
    --ft_type ln
done;


for PRED_LEN in 96 192 336 720; do
    CUDA_VISIBLE_DEVICES=6 python -u run.py \
    --task_name long_term_forecast \
    --learning_rate 0.0001 \
    --batch_size 32 \
    --is_training 1 \
    --model VisionTS \
    --root_path ./dataset/ETT-small/ \
    --data_path ETTm2.csv \
    --save_dir save/ETTm2_RQ8_$PRED_LEN \
    --model_id VisionTS_ETTm2_$PRED_LEN \
    --data ETTm2 \
    --features M \
    --train_epochs 30 \
    --lradj fixed \
    --ckpt $ckpt \
    --vm_arch $vm_arch \
    --seq_len $context_len \
    --periodicity 192 \
    --pred_len $PRED_LEN \
    --align_const $align_const \
    --norm_const $norm_const \
    --ft_type ln
done;