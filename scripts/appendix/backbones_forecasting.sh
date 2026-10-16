# Appendix "Comparison with SOTA LVMs" (forecasting): encoder + linear head (UTSRegression).
source scripts/appendix/common.sh

for row in "${TSF_DATASETS[@]}"; do
  IFS='|' read -r name data root file period ctx nvars horizons <<< "$row"
  for backbone_dim in vit_base:768 dinov2:768 mlcd:768 sam:256; do
    backbone=${backbone_dim%%:*}; vm_dim=${backbone_dim##*:}
    for pl in $horizons; do
      python -u run.py --task_name long_term_forecast --is_training 1 --model UTSRegression \
        --root_path $root --data_path $file --data $data --features M \
        --model_id ${name}_${backbone}_$pl --save_dir save/${name}_${backbone}_$pl \
        --seq_len $ctx --context_len $ctx --pred_len $pl --periodicity $period --num_features $nvars \
        --backbone $backbone --vm_dim $vm_dim --head_dim 128 --head_dropout 0.3 \
        --learning_rate 0.0001 --batch_size 32 --train_epochs 20 --patience 3 --lradj fixed
    done
  done
done
