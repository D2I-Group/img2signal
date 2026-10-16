# Appendix "Robustness of MVH Imaging to Variate Permutation": five random variate orders (seeds 1-5).
# --variate_perm 0 keeps the native order.
source scripts/appendix/common.sh

for seed in 1 2 3 4 5; do
  for cfg in SpokenArabicDigits:16:0.0001 UWaveGestureLibrary:32:0.0001 Handwriting:32:0.00001; do
    IFS=':' read -r ds bs lr <<< "$cfg"
    python -u run.py --task_name classification --is_training 1 --model UTSClassification --data UEA \
      --root_path ./dataset/$ds/ --model_id ${ds}_MVH_perm$seed --time2img MVH --variate_perm $seed \
      --backbone vit_base --resize default --ft_type full --lradj fixed \
      --learning_rate $lr --batch_size $bs --train_epochs 30 --patience 15
  done

  for row in "${TSF_DATASETS[@]}"; do
    IFS='|' read -r name data root file period ctx nvars horizons <<< "$row"
    align=1; [ "$name" = ETTh1 ] && align=0.4
    [ "$data" = custom ] && data=customMVH
    for pl in $horizons; do
      python -u run.py --task_name long_term_forecast --is_training 1 --model MVH_RQ3 \
        --root_path $root --data_path $file --data $data --features M \
        --model_id ${name}_MVH_perm${seed}_$pl --save_dir save/${name}_MVH_perm${seed}_$pl \
        --seq_len $ctx --pred_len $pl --num_features $nvars --norm_const 1 --align_const $align \
        --vm_arch mae_base --ft_type ln --variate_perm $seed --lradj fixed \
        --learning_rate 0.0001 --batch_size 32 --train_epochs 1 --patience 1
    done
  done
done
