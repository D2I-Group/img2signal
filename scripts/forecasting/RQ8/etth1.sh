export ckpt="./ckpt/"
export vm_arch="mae_base"
export context_len=336
export align_const=0.4
export norm_const=0.4


for PRED_LEN in 96 192 336 720; do
    CUDA_VISIBLE_DEVICES=4 python -u run.py \
    --task_name long_term_forecast \
    --learning_rate 0.0001 \
    --batch_size 32 \
    --is_training 1 \
    --model VisionTS \
    --root_path ./dataset/ETT-small/ \
    --data_path ETTh1.csv \
    --save_dir save/ETTh1_RQ8_$PRED_LEN \
    --model_id VisionTS_ETTh1_$PRED_LEN \
    --data ETTh1 \
    --features M \
    --train_epochs 20 \
    --ckpt $ckpt \
    --lradj fixed \
    --vm_arch $vm_arch \
    --seq_len $context_len \
    --periodicity 4 \
    --pred_len $PRED_LEN \
    --align_const $align_const \
    --norm_const $norm_const \
    --ft_type ln
done;



for PRED_LEN in 96 192 336 720; do
    CUDA_VISIBLE_DEVICES=4 python -u run.py \
    --task_name long_term_forecast \
    --learning_rate 0.0001 \
    --batch_size 32 \
    --is_training 1 \
    --model VisionTS \
    --root_path ./dataset/ETT-small/ \
    --data_path ETTh1.csv \
    --save_dir save/ETTh1_RQ8_$PRED_LEN \
    --model_id VisionTS_ETTh1_$PRED_LEN \
    --data ETTh1 \
    --features M \
    --train_epochs 20 \
    --ckpt $ckpt \
    --lradj fixed \
    --vm_arch $vm_arch \
    --seq_len $context_len \
    --periodicity 8 \
    --pred_len $PRED_LEN \
    --align_const $align_const \
    --norm_const $norm_const \
    --ft_type ln
done;




for PRED_LEN in 96 192 336 720; do
    CUDA_VISIBLE_DEVICES=4 python -u run.py \
    --task_name long_term_forecast \
    --learning_rate 0.0001 \
    --batch_size 32 \
    --is_training 1 \
    --model VisionTS \
    --root_path ./dataset/ETT-small/ \
    --data_path ETTh1.csv \
    --save_dir save/ETTh1_RQ8_$PRED_LEN \
    --model_id VisionTS_ETTh1_$PRED_LEN \
    --data ETTh1 \
    --features M \
    --train_epochs 20 \
    --ckpt $ckpt \
    --lradj fixed \
    --vm_arch $vm_arch \
    --seq_len $context_len \
    --periodicity 12 \
    --pred_len $PRED_LEN \
    --align_const $align_const \
    --norm_const $norm_const \
    --ft_type ln
done;


for PRED_LEN in 96 192 336 720; do
    CUDA_VISIBLE_DEVICES=4 python -u run.py \
    --task_name long_term_forecast \
    --learning_rate 0.0001 \
    --batch_size 32 \
    --is_training 1 \
    --model VisionTS \
    --root_path ./dataset/ETT-small/ \
    --data_path ETTh1.csv \
    --save_dir save/ETTh1_RQ8_$PRED_LEN \
    --model_id VisionTS_ETTh1_$PRED_LEN \
    --data ETTh1 \
    --features M \
    --train_epochs 20 \
    --ckpt $ckpt \
    --lradj fixed \
    --vm_arch $vm_arch \
    --seq_len $context_len \
    --periodicity 16 \
    --pred_len $PRED_LEN \
    --align_const $align_const \
    --norm_const $norm_const \
    --ft_type ln
done;


for PRED_LEN in 96 192 336 720; do
    CUDA_VISIBLE_DEVICES=4 python -u run.py \
    --task_name long_term_forecast \
    --learning_rate 0.0001 \
    --batch_size 32 \
    --is_training 1 \
    --model VisionTS \
    --root_path ./dataset/ETT-small/ \
    --data_path ETTh1.csv \
    --save_dir save/ETTh1_RQ8_$PRED_LEN \
    --model_id VisionTS_ETTh1_$PRED_LEN \
    --data ETTh1 \
    --features M \
    --train_epochs 20 \
    --ckpt $ckpt \
    --lradj fixed \
    --vm_arch $vm_arch \
    --seq_len $context_len \
    --periodicity 20 \
    --pred_len $PRED_LEN \
    --align_const $align_const \
    --norm_const $norm_const \
    --ft_type ln
done;



for PRED_LEN in 96 192 336 720; do
    CUDA_VISIBLE_DEVICES=4 python -u run.py \
    --task_name long_term_forecast \
    --learning_rate 0.0001 \
    --batch_size 32 \
    --is_training 1 \
    --model VisionTS \
    --root_path ./dataset/ETT-small/ \
    --data_path ETTh1.csv \
    --save_dir save/ETTh1_RQ8_$PRED_LEN \
    --model_id VisionTS_ETTh1_$PRED_LEN \
    --data ETTh1 \
    --features M \
    --train_epochs 20 \
    --ckpt $ckpt \
    --lradj fixed \
    --vm_arch $vm_arch \
    --seq_len $context_len \
    --periodicity 28 \
    --pred_len $PRED_LEN \
    --align_const $align_const \
    --norm_const $norm_const \
    --ft_type ln
done;



for PRED_LEN in 96 192 336 720; do
    CUDA_VISIBLE_DEVICES=4 python -u run.py \
    --task_name long_term_forecast \
    --learning_rate 0.0001 \
    --batch_size 32 \
    --is_training 1 \
    --model VisionTS \
    --root_path ./dataset/ETT-small/ \
    --data_path ETTh1.csv \
    --save_dir save/ETTh1_RQ8_$PRED_LEN \
    --model_id VisionTS_ETTh1_$PRED_LEN \
    --data ETTh1 \
    --features M \
    --train_epochs 20 \
    --ckpt $ckpt \
    --lradj fixed \
    --vm_arch $vm_arch \
    --seq_len $context_len \
    --periodicity 32 \
    --pred_len $PRED_LEN \
    --align_const $align_const \
    --norm_const $norm_const \
    --ft_type ln
done;



for PRED_LEN in 96 192 336 720; do
    CUDA_VISIBLE_DEVICES=4 python -u run.py \
    --task_name long_term_forecast \
    --learning_rate 0.0001 \
    --batch_size 32 \
    --is_training 1 \
    --model VisionTS \
    --root_path ./dataset/ETT-small/ \
    --data_path ETTh1.csv \
    --save_dir save/ETTh1_RQ8_$PRED_LEN \
    --model_id VisionTS_ETTh1_$PRED_LEN \
    --data ETTh1 \
    --features M \
    --train_epochs 20 \
    --ckpt $ckpt \
    --lradj fixed \
    --vm_arch $vm_arch \
    --seq_len $context_len \
    --periodicity 36 \
    --pred_len $PRED_LEN \
    --align_const $align_const \
    --norm_const $norm_const \
    --ft_type ln
done;




for PRED_LEN in 96 192 336 720; do
    CUDA_VISIBLE_DEVICES=4 python -u run.py \
    --task_name long_term_forecast \
    --learning_rate 0.0001 \
    --batch_size 32 \
    --is_training 1 \
    --model VisionTS \
    --root_path ./dataset/ETT-small/ \
    --data_path ETTh1.csv \
    --save_dir save/ETTh1_RQ8_$PRED_LEN \
    --model_id VisionTS_ETTh1_$PRED_LEN \
    --data ETTh1 \
    --features M \
    --train_epochs 20 \
    --ckpt $ckpt \
    --lradj fixed \
    --vm_arch $vm_arch \
    --seq_len $context_len \
    --periodicity 40 \
    --pred_len $PRED_LEN \
    --align_const $align_const \
    --norm_const $norm_const \
    --ft_type ln
done;



for PRED_LEN in 96 192 336 720; do
    CUDA_VISIBLE_DEVICES=4 python -u run.py \
    --task_name long_term_forecast \
    --learning_rate 0.0001 \
    --batch_size 32 \
    --is_training 1 \
    --model VisionTS \
    --root_path ./dataset/ETT-small/ \
    --data_path ETTh1.csv \
    --save_dir save/ETTh1_RQ8_$PRED_LEN \
    --model_id VisionTS_ETTh1_$PRED_LEN \
    --data ETTh1 \
    --features M \
    --train_epochs 20 \
    --ckpt $ckpt \
    --lradj fixed \
    --vm_arch $vm_arch \
    --seq_len $context_len \
    --periodicity 44 \
    --pred_len $PRED_LEN \
    --align_const $align_const \
    --norm_const $norm_const \
    --ft_type ln
done;



for PRED_LEN in 96 192 336 720; do
    CUDA_VISIBLE_DEVICES=4 python -u run.py \
    --task_name long_term_forecast \
    --learning_rate 0.0001 \
    --batch_size 32 \
    --is_training 1 \
    --model VisionTS \
    --root_path ./dataset/ETT-small/ \
    --data_path ETTh1.csv \
    --save_dir save/ETTh1_RQ8_$PRED_LEN \
    --model_id VisionTS_ETTh1_$PRED_LEN \
    --data ETTh1 \
    --features M \
    --train_epochs 20 \
    --ckpt $ckpt \
    --lradj fixed \
    --vm_arch $vm_arch \
    --seq_len $context_len \
    --periodicity 48 \
    --pred_len $PRED_LEN \
    --align_const $align_const \
    --norm_const $norm_const \
    --ft_type ln
done;