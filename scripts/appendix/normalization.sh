# Appendix "Evaluation about Normalization Methods" (MAE + UVH forecasting).
#   none          : --dataset_norm 0 --norm_mode dataset
#   dataset only  : --dataset_norm 1 --norm_mode dataset
#   instance only : --dataset_norm 0 --norm_mode instance
#   both (default): --dataset_norm 1 --norm_mode instance
# Metrics are always reported in the train-standardized space.
source scripts/appendix/common.sh

for row in "${TSF_DATASETS[@]}"; do
  IFS='|' read -r name data root file period ctx nvars horizons <<< "$row"
  for setting in none:0:dataset dataset:1:dataset instance:0:instance both:1:instance; do
    IFS=':' read -r tag dnorm nmode <<< "$setting"
    for pl in $horizons; do
      python -u run.py --task_name long_term_forecast --is_training 1 --model VisionTS \
        --root_path $root --data_path $file --data $data --features M \
        --model_id ${name}_norm_${tag}_$pl --save_dir save/${name}_norm_${tag}_$pl \
        --seq_len $ctx --pred_len $pl --periodicity $period --norm_const 0.4 --align_const 0.4 \
        --vm_arch mae_base --ft_type ln --dataset_norm $dnorm --norm_mode $nmode \
        --learning_rate 0.0001 --batch_size 32 --train_epochs 1 --patience 1
    done
  done
done
