source scripts/appendix/common.sh

for ds in $TSC_DATASETS; do
  bs=32; [ "$ds" = FaceDetection ] && bs=4
  python -u run.py --task_name classification --is_training 1 --model UTSClassification --data UEA \
    --root_path ./dataset/$ds/ --model_id ${ds}_GAF_lora --time2img GAF --backbone vit_base --resize default \
    --ft_type lora --lora_r 8 --lora_alpha 16 --learning_rate 0.0001 --batch_size $bs --train_epochs 5 --patience 5
done

for row in "${TSF_DATASETS[@]}"; do
  IFS='|' read -r name data root file period ctx nvars horizons <<< "$row"
  for pl in $horizons; do
    python -u run.py --task_name long_term_forecast --is_training 1 --model VisionTS \
      --root_path $root --data_path $file --data $data --features M \
      --model_id ${name}_lora_$pl --save_dir save/${name}_lora_$pl \
      --seq_len $ctx --pred_len $pl --periodicity $period --norm_const 0.4 --align_const 0.4 \
      --vm_arch mae_base --ft_type lora --lora_r 8 --lora_alpha 16 \
      --learning_rate 0.0001 --batch_size 32 --train_epochs 5 --patience 5
  done
done
