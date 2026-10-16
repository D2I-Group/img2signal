
CUDA_VISIBLE_DEVICES=6 python -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path ./dataset/Heartbeat/ \
  --model_id Heartbeat \
  --model UTSClassification \
  --data UEA \
  --batch_size 4 \
  --learning_rate 0.0001 \
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