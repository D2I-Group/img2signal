# Appendix "Evaluation of LVMs on Datasets with Weak Periodicity": MAE (UVH) vs. PatchTST on Exchange and Solar.
# Prepare Solar first:  python tools/prepare_solar.py --src ./dataset/solar/solar_AL.txt --dst ./dataset/solar/solar.csv
source scripts/appendix/common.sh

# name|root_path|data_path|periodicity|n_variates|MAE batch size
for row in "Exchange|./dataset/exchange_rate/|exchange_rate.csv|30|8|64" "Solar|./dataset/solar/|solar.csv|24|137|4"; do
  IFS='|' read -r name root file period nvars bs <<< "$row"
  for pl in 96 192 336 720; do
    python -u run.py --task_name long_term_forecast --is_training 1 --model VisionTS \
      --root_path $root --data_path $file --data custom --features M \
      --model_id ${name}_MAE_$pl --save_dir save/${name}_MAE_$pl \
      --seq_len 336 --pred_len $pl --periodicity $period --norm_const 0.4 --align_const 0.4 \
      --vm_arch mae_base --ft_type ln --learning_rate 0.00005 --batch_size $bs --train_epochs 10 --patience 3

    python -u run.py --task_name long_term_forecast --is_training 1 --model PatchTST \
      --root_path $root --data_path $file --data custom --features M \
      --model_id ${name}_PatchTST_$pl --save_dir save/${name}_PatchTST_$pl \
      --seq_len 336 --label_len 48 --pred_len $pl --enc_in $nvars --dec_in $nvars --c_out $nvars \
      --e_layers 3 --n_heads 16 --d_model 128 --d_ff 256 --dropout 0.2 \
      --learning_rate 0.0001 --batch_size 128 --train_epochs 100 --patience 10
  done
done
