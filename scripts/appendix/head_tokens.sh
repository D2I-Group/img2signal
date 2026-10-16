source scripts/appendix/common.sh

for ds in UWaveGestureLibrary Handwriting; do
  for tokens in all cls mean; do
    python -u run.py --task_name classification --is_training 1 --model UTSClassification --data UEA \
      --root_path ./dataset/$ds/ --model_id ${ds}_GAF_head_$tokens --time2img GAF --backbone vit_base --resize default \
      --ft_type full --head_tokens $tokens --learning_rate 0.0001 --batch_size 32 --train_epochs 30 --patience 8
  done
done
