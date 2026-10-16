export ckpt="./ckpt/"
export vm_arch="mae_base"
export context_len=104
export periodicity=52
export align_const=0.4
export norm_const=1.0


# for PRED_LEN in 24 36 48 60; do
#     CUDA_VISIBLE_DEVICES=0 python -u run.py \
#     --task_name long_term_forecast \
#     --learning_rate 0.00001 \
#     --batch_size 32 \
#     --is_training 1 \
#     --model VisionTS \
#     --root_path ./dataset/illness/ \
#     --data_path national_illness.csv \
#     --save_dir save/illness_RQ4_$PRED_LEN \
#     --model_id VisionTS_illness_$PRED_LEN \
#     --data custom \
#     --features M \
#     --train_epochs 100 \
#     --lradj fixed \
#     --patience 5 \
#     --ckpt $ckpt \
#     --vm_arch $vm_arch \
#     --seq_len $context_len \
#     --periodicity $periodicity \
#     --pred_len $PRED_LEN \
#     --align_const $align_const \
#     --norm_const $norm_const \
#     --ft_type full
# done;



# for PRED_LEN in 24 36 48 60; do
#     CUDA_VISIBLE_DEVICES=0 python -u run.py \
#     --task_name long_term_forecast \
#     --learning_rate 0.00001 \
#     --batch_size 32 \
#     --is_training 1 \
#     --model VisionTS \
#     --root_path ./dataset/illness/ \
#     --data_path national_illness.csv \
#     --save_dir save/illness_RQ4_$PRED_LEN \
#     --model_id VisionTS_illness_$PRED_LEN \
#     --data custom \
#     --features M \
#     --train_epochs 100 \
#     --lradj fixed \
#     --patience 5 \
#     --ckpt $ckpt \
#     --vm_arch $vm_arch \
#     --seq_len $context_len \
#     --periodicity $periodicity \
#     --pred_len $PRED_LEN \
#     --align_const $align_const \
#     --norm_const $norm_const \
#     --ft_type full_wo_cls_mask
# done;



for PRED_LEN in 24 36 48 60; do
    CUDA_VISIBLE_DEVICES=0 python -u run.py \
    --task_name long_term_forecast \
    --learning_rate 0.00001 \
    --batch_size 32 \
    --is_training 1 \
    --model VisionTS \
    --root_path ./dataset/illness/ \
    --data_path national_illness.csv \
    --save_dir save/illness_RQ4_$PRED_LEN \
    --model_id VisionTS_illness_$PRED_LEN \
    --data custom \
    --features M \
    --train_epochs 100 \
    --lradj fixed \
    --patience 5 \
    --ckpt $ckpt \
    --vm_arch $vm_arch \
    --seq_len $context_len \
    --periodicity $periodicity \
    --pred_len $PRED_LEN \
    --align_const $align_const \
    --norm_const $norm_const \
    --ft_type ln
done;




# for PRED_LEN in 24 36 48 60; do
#     CUDA_VISIBLE_DEVICES=0 python -u run.py \
#     --task_name long_term_forecast \
#     --learning_rate 0.00001 \
#     --batch_size 32 \
#     --is_training 1 \
#     --model VisionTS \
#     --root_path ./dataset/illness/ \
#     --data_path national_illness.csv \
#     --save_dir save/illness_RQ4_$PRED_LEN \
#     --model_id VisionTS_illness_$PRED_LEN \
#     --data custom \
#     --features M \
#     --train_epochs 100 \
#     --lradj fixed \
#     --patience 5 \
#     --ckpt $ckpt \
#     --vm_arch $vm_arch \
#     --seq_len $context_len \
#     --periodicity $periodicity \
#     --pred_len $PRED_LEN \
#     --align_const $align_const \
#     --norm_const $norm_const \
#     --ft_type mlp+ln
# done;



# for PRED_LEN in 24 36 48 60; do
#     CUDA_VISIBLE_DEVICES=7 python -u run.py \
#     --task_name long_term_forecast \
#     --learning_rate 0.00001 \
#     --batch_size 32 \
#     --is_training 1 \
#     --model VisionTS \
#     --root_path ./dataset/illness/ \
#     --data_path national_illness.csv \
#     --save_dir save/illness_RQ4_$PRED_LEN \
#     --model_id VisionTS_illness_$PRED_LEN \
#     --data custom \
#     --features M \
#     --train_epochs 0 \
#     --lradj fixed \
#     --patience 5 \
#     --ckpt $ckpt \
#     --vm_arch $vm_arch \
#     --seq_len $context_len \
#     --periodicity $periodicity \
#     --pred_len $PRED_LEN \
#     --align_const $align_const \
#     --norm_const $norm_const \
#     --ft_type zero-shot
# done;



# for PRED_LEN in 24 36 48 60; do
#     CUDA_VISIBLE_DEVICES=0 python -u run.py \
#     --task_name long_term_forecast \
#     --learning_rate 0.00001 \
#     --batch_size 32 \
#     --is_training 1 \
#     --model VisionTS \
#     --root_path ./dataset/illness/ \
#     --data_path national_illness.csv \
#     --save_dir save/illness_RQ4_$PRED_LEN \
#     --model_id VisionTS_illness_$PRED_LEN \
#     --data custom \
#     --features M \
#     --train_epochs 100 \
#     --lradj fixed \
#     --patience 5 \
#     --no_ckpt \
#     --ckpt $ckpt \
#     --vm_arch $vm_arch \
#     --seq_len $context_len \
#     --periodicity $periodicity \
#     --pred_len $PRED_LEN \
#     --align_const $align_const \
#     --norm_const $norm_const \
#     --ft_type full
# done;
