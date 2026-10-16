
CUDA_VISIBLE_DEVICES=0 python -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path ./dataset/EthanolConcentration/ \
  --model_id EthanolConcentrationRP \
  --model UTSClassification \
  --data UEA \
  --batch_size 32 \
  --learning_rate 0.0001 \
  --train_epochs 30 \
  --time2img RP \
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


CUDA_VISIBLE_DEVICES=0 python -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path ./dataset/JapaneseVowels/ \
  --model_id JapaneseVowels_RP \
  --model UTSClassification \
  --data UEA \
  --batch_size 32 \
  --learning_rate 0.0001 \
  --train_epochs 30 \
  --time2img RP \
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




CUDA_VISIBLE_DEVICES=0 python -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path ./dataset/Handwriting/ \
  --model_id Handwriting_RP \
  --model UTSClassification \
  --data UEA \
  --batch_size 32 \
  --learning_rate 0.00001 \
  --train_epochs 30 \
  --time2img RP \
  --lradj fixed \
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




CUDA_VISIBLE_DEVICES=0 python -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path ./dataset/Heartbeat/ \
  --model_id Heartbeat_RP \
  --model UTSClassification \
  --data UEA \
  --batch_size 4 \
  --learning_rate 0.0001 \
  --train_epochs 30 \
  --time2img RP \
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


CUDA_VISIBLE_DEVICES=0 python -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path ./dataset/SelfRegulationSCP1/ \
  --model_id SelfRegulationSCP1_RP \
  --model UTSClassification \
  --data UEA \
  --batch_size 32 \
  --learning_rate 0.0001 \
  --lradj fixed \
  --train_epochs 30 \
  --time2img RP \
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


CUDA_VISIBLE_DEVICES=0 python -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path ./dataset/SelfRegulationSCP2/ \
  --model_id SelfRegulationSCP2_RP \
  --model UTSClassification \
  --data UEA \
  --batch_size 32 \
  --learning_rate 0.0001 \
  --train_epochs 30 \
  --time2img RP \
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


CUDA_VISIBLE_DEVICES=0 python -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path ./dataset/SpokenArabicDigits/ \
  --model_id SpokenArabicDigits_RP \
  --model UTSClassification \
  --data UEA \
  --batch_size 16 \
  --learning_rate 0.0001 \
  --train_epochs 30 \
  --time2img RP \
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


CUDA_VISIBLE_DEVICES=0 python -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path ./dataset/UWaveGestureLibrary/ \
  --model_id UWaveGestureLibrary_RP \
  --model UTSClassification \
  --data UEA \
  --batch_size 32 \
  --learning_rate 0.0001 \
  --train_epochs 30 \
  --time2img RP \
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