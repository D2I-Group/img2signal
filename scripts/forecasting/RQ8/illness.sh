export ckpt="./ckpt/"
export vm_arch="mae_base"
export context_len=104
export align_const=0.4
export norm_const=1.0

# for PRED_LEN in 24 36 48 60; do
#     CUDA_VISIBLE_DEVICES=1 python -u run.py \
#     --task_name long_term_forecast \
#     --learning_rate 0.00001 \
#     --batch_size 32 \
#     --is_training 1 \
#     --model VisionTS \
#     --root_path ./dataset/illness/ \
#     --data_path national_illness.csv \
#     --save_dir save/illness_RQ7_$PRED_LEN \
#     --model_id VisionTS_illness_$PRED_LEN \
#     --data custom \
#     --features M \
#     --train_epochs 100 \
#     --lradj fixed \
#     --patience 10 \
#     --ckpt $ckpt \
#     --vm_arch $vm_arch \
#     --seq_len $context_len \
#     --periodicity 8 \
#     --pred_len $PRED_LEN \
#     --align_const $align_const \
#     --ft_type ln \
#     --norm_const $norm_const
# done;

# for PRED_LEN in 24 36 48 60; do
#     CUDA_VISIBLE_DEVICES=1 python -u run.py \
#     --task_name long_term_forecast \
#     --learning_rate 0.00001 \
#     --batch_size 32 \
#     --is_training 1 \
#     --model VisionTS \
#     --root_path ./dataset/illness/ \
#     --data_path national_illness.csv \
#     --save_dir save/illness_RQ7_$PRED_LEN \
#     --model_id VisionTS_illness_$PRED_LEN \
#     --data custom \
#     --features M \
#     --train_epochs 100 \
#     --lradj fixed \
#     --patience 10 \
#     --ckpt $ckpt \
#     --vm_arch $vm_arch \
#     --seq_len $context_len \
#     --periodicity 18 \
#     --pred_len $PRED_LEN \
#     --align_const $align_const \
#     --ft_type ln \
#     --norm_const $norm_const
# done;


# for PRED_LEN in 24 36 48 60; do
#     CUDA_VISIBLE_DEVICES=1 python -u run.py \
#     --task_name long_term_forecast \
#     --learning_rate 0.00001 \
#     --batch_size 32 \
#     --is_training 1 \
#     --model VisionTS \
#     --root_path ./dataset/illness/ \
#     --data_path national_illness.csv \
#     --save_dir save/illness_RQ7_$PRED_LEN \
#     --model_id VisionTS_illness_$PRED_LEN \
#     --data custom \
#     --features M \
#     --train_epochs 100 \
#     --lradj fixed \
#     --patience 10 \
#     --ckpt $ckpt \
#     --vm_arch $vm_arch \
#     --seq_len $context_len \
#     --periodicity 26 \
#     --pred_len $PRED_LEN \
#     --align_const $align_const \
#     --ft_type ln \
#     --norm_const $norm_const
# done;


# for PRED_LEN in 24 36 48 60; do
#     CUDA_VISIBLE_DEVICES=1 python -u run.py \
#     --task_name long_term_forecast \
#     --learning_rate 0.00001 \
#     --batch_size 32 \
#     --is_training 1 \
#     --model VisionTS \
#     --root_path ./dataset/illness/ \
#     --data_path national_illness.csv \
#     --save_dir save/illness_RQ7_$PRED_LEN \
#     --model_id VisionTS_illness_$PRED_LEN \
#     --data custom \
#     --features M \
#     --train_epochs 100 \
#     --lradj fixed \
#     --patience 10 \
#     --ckpt $ckpt \
#     --vm_arch $vm_arch \
#     --seq_len $context_len \
#     --periodicity 34 \
#     --pred_len $PRED_LEN \
#     --align_const $align_const \
#     --ft_type ln \
#     --norm_const $norm_const
# done;


# for PRED_LEN in 24 36 48 60; do
#     CUDA_VISIBLE_DEVICES=1 python -u run.py \
#     --task_name long_term_forecast \
#     --learning_rate 0.00001 \
#     --batch_size 32 \
#     --is_training 1 \
#     --model VisionTS \
#     --root_path ./dataset/illness/ \
#     --data_path national_illness.csv \
#     --save_dir save/illness_RQ7_$PRED_LEN \
#     --model_id VisionTS_illness_$PRED_LEN \
#     --data custom \
#     --features M \
#     --train_epochs 100 \
#     --lradj fixed \
#     --patience 10 \
#     --ckpt $ckpt \
#     --vm_arch $vm_arch \
#     --seq_len $context_len \
#     --periodicity 43 \
#     --pred_len $PRED_LEN \
#     --align_const $align_const \
#     --ft_type ln \
#     --norm_const $norm_const
# done;


# for PRED_LEN in 24 36 48 60; do
#     CUDA_VISIBLE_DEVICES=1 python -u run.py \
#     --task_name long_term_forecast \
#     --learning_rate 0.00001 \
#     --batch_size 32 \
#     --is_training 1 \
#     --model VisionTS \
#     --root_path ./dataset/illness/ \
#     --data_path national_illness.csv \
#     --save_dir save/illness_RQ7_$PRED_LEN \
#     --model_id VisionTS_illness_$PRED_LEN \
#     --data custom \
#     --features M \
#     --train_epochs 100 \
#     --lradj fixed \
#     --patience 10 \
#     --ckpt $ckpt \
#     --vm_arch $vm_arch \
#     --seq_len $context_len \
#     --periodicity 52 \
#     --pred_len $PRED_LEN \
#     --align_const $align_const \
#     --ft_type ln \
#     --norm_const $norm_const
# done;

# for PRED_LEN in 24 36 48 60; do
#     CUDA_VISIBLE_DEVICES=1 python -u run.py \
#     --task_name long_term_forecast \
#     --learning_rate 0.00001 \
#     --batch_size 32 \
#     --is_training 1 \
#     --model VisionTS \
#     --root_path ./dataset/illness/ \
#     --data_path national_illness.csv \
#     --save_dir save/illness_RQ7_$PRED_LEN \
#     --model_id VisionTS_illness_$PRED_LEN \
#     --data custom \
#     --features M \
#     --train_epochs 100 \
#     --lradj fixed \
#     --patience 10 \
#     --ckpt $ckpt \
#     --vm_arch $vm_arch \
#     --seq_len $context_len \
#     --periodicity 60 \
#     --pred_len $PRED_LEN \
#     --align_const $align_const \
#     --ft_type ln \
#     --norm_const $norm_const
# done;



# for PRED_LEN in 24 36 48 60; do
#     CUDA_VISIBLE_DEVICES=1 python -u run.py \
#     --task_name long_term_forecast \
#     --learning_rate 0.00001 \
#     --batch_size 32 \
#     --is_training 1 \
#     --model VisionTS \
#     --root_path ./dataset/illness/ \
#     --data_path national_illness.csv \
#     --save_dir save/illness_RQ7_$PRED_LEN \
#     --model_id VisionTS_illness_$PRED_LEN \
#     --data custom \
#     --features M \
#     --train_epochs 100 \
#     --lradj fixed \
#     --patience 10 \
#     --ckpt $ckpt \
#     --vm_arch $vm_arch \
#     --seq_len $context_len \
#     --periodicity 69 \
#     --pred_len $PRED_LEN \
#     --align_const $align_const \
#     --ft_type ln \
#     --norm_const $norm_const
# done;



# for PRED_LEN in 24 36 48 60; do
#     CUDA_VISIBLE_DEVICES=1 python -u run.py \
#     --task_name long_term_forecast \
#     --learning_rate 0.00001 \
#     --batch_size 32 \
#     --is_training 1 \
#     --model VisionTS \
#     --root_path ./dataset/illness/ \
#     --data_path national_illness.csv \
#     --save_dir save/illness_RQ7_$PRED_LEN \
#     --model_id VisionTS_illness_$PRED_LEN \
#     --data custom \
#     --features M \
#     --train_epochs 100 \
#     --lradj fixed \
#     --patience 10 \
#     --ckpt $ckpt \
#     --vm_arch $vm_arch \
#     --seq_len $context_len \
#     --periodicity 78 \
#     --pred_len $PRED_LEN \
#     --align_const $align_const \
#     --ft_type ln \
#     --norm_const $norm_const
# done;



# for PRED_LEN in 24 36 48 60; do
#     CUDA_VISIBLE_DEVICES=1 python -u run.py \
#     --task_name long_term_forecast \
#     --learning_rate 0.00001 \
#     --batch_size 32 \
#     --is_training 1 \
#     --model VisionTS \
#     --root_path ./dataset/illness/ \
#     --data_path national_illness.csv \
#     --save_dir save/illness_RQ7_$PRED_LEN \
#     --model_id VisionTS_illness_$PRED_LEN \
#     --data custom \
#     --features M \
#     --train_epochs 100 \
#     --lradj fixed \
#     --patience 10 \
#     --ckpt $ckpt \
#     --vm_arch $vm_arch \
#     --seq_len $context_len \
#     --periodicity 87 \
#     --pred_len $PRED_LEN \
#     --align_const $align_const \
#     --ft_type ln \
#     --norm_const $norm_const
# done;



for PRED_LEN in 24 36 48 60; do
    CUDA_VISIBLE_DEVICES=1 python -u run.py \
    --task_name long_term_forecast \
    --learning_rate 0.00001 \
    --batch_size 32 \
    --is_training 1 \
    --model VisionTS \
    --root_path ./dataset/illness/ \
    --data_path national_illness.csv \
    --save_dir save/illness_RQ7_$PRED_LEN \
    --model_id VisionTS_illness_$PRED_LEN \
    --data custom \
    --features M \
    --train_epochs 100 \
    --lradj fixed \
    --patience 10 \
    --ckpt $ckpt \
    --vm_arch $vm_arch \
    --seq_len $context_len \
    --periodicity 95 \
    --pred_len $PRED_LEN \
    --align_const $align_const \
    --ft_type ln \
    --norm_const $norm_const
done;


for PRED_LEN in 24 36 48 60; do
    CUDA_VISIBLE_DEVICES=1 python -u run.py \
    --task_name long_term_forecast \
    --learning_rate 0.00001 \
    --batch_size 32 \
    --is_training 1 \
    --model VisionTS \
    --root_path ./dataset/illness/ \
    --data_path national_illness.csv \
    --save_dir save/illness_RQ7_$PRED_LEN \
    --model_id VisionTS_illness_$PRED_LEN \
    --data custom \
    --features M \
    --train_epochs 100 \
    --lradj fixed \
    --patience 10 \
    --ckpt $ckpt \
    --vm_arch $vm_arch \
    --seq_len $context_len \
    --periodicity 104 \
    --pred_len $PRED_LEN \
    --align_const $align_const \
    --ft_type ln \
    --norm_const $norm_const
done;