# Appendix "Comparison with SOTA LVMs" (classification): DINOv2, MLCD, Qwen2-VL vision tower, SAM.
# Weights are downloaded from timm / Hugging Face on first use.
source scripts/appendix/common.sh

run() {  # dataset backbone lr batch ft_type resize
  python -u run.py --task_name classification --is_training 1 --model UTSClassification --data UEA \
    --root_path ./dataset/$1/ --model_id ${1}_GAF_$2 --time2img GAF --backbone $2 \
    --learning_rate $3 --batch_size $4 --ft_type $5 --resize $6 --train_epochs 30 --patience 8
}

for ds in $TSC_DATASETS; do
  run $ds dinov2 1e-5 8 full default
  run $ds mlcd   1e-5 8 full default
  run $ds qwenvl 3e-5 8 full default
  run $ds sam    1e-5 1 ln   sam      # SAM runs at its native 1024x1024 resolution
done
