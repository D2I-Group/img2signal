
CUDA_VISIBLE_DEVICES=6 python -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path ./dataset/SelfRegulationSCP1/ \
  --model_id SelfRegulationSCP1 \
  --model UTSClassification \
  --data UEA \
  --batch_size 32 \
  --learning_rate 0.0001 \
  --lradj fixed \
  --train_epochs 30 \
  --time2img GAF \
  --backbone vit_base \
  --resize default \
  --patience 15


echo "  ########################################################"
echo "  #    ######      ########    ##     ##    ########     #"
echo "  #    ##     ##   ##     ##   ###    ##    ##           #"
echo "  #    ##      ##  ##     ##   ## ##  ##    ##           #"
echo "  #    ##       ## ##     ##   ##  ## ##    ######       #"
echo "  #    ##       ## ##     ##   ##   ####    ##           #"
echo "  #    ##      ##  ##     ##   ##    ###    ##           #"
echo "  #    ##     ##   ##     ##   ##     ##    ##           #"
echo "  #    ######      ########    ##     ##    ########     #"
echo "  ########################################################"