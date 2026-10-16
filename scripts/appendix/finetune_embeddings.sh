source scripts/appendix/common.sh

for ft in pos_embed patch_embed; do
  for ds in $TSC_DATASETS; do
    python -u run.py --task_name classification --is_training 1 --model UTSClassification --data UEA \
      --root_path ./dataset/$ds/ --model_id ${ds}_GAF_$ft --time2img GAF --backbone vit_base --resize default \
      --ft_type $ft --learning_rate 0.0001 --batch_size 32 --train_epochs 30 --patience 8
  done
  for row in "${TSF_DATASETS[@]}"; do
    IFS='|' read -r name data root file period ctx nvars horizons <<< "$row"
    for pl in $horizons; do
      python -u run.py --task_name long_term_forecast --is_training 1 --model VisionTS \
        --root_path $root --data_path $file --data $data --features M \
        --model_id ${name}_${ft}_$pl --save_dir save/${name}_${ft}_$pl \
        --seq_len $ctx --pred_len $pl --periodicity $period --norm_const 0.4 --align_const 0.4 \
        --vm_arch mae_base --ft_type $ft --learning_rate 0.0001 --batch_size 32 --train_epochs 20 --patience 3 --lradj fixed
    done
  done
done
