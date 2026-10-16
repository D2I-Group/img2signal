# CUDA_VISIBLE_DEVICES=1 python -u run.py \
#   --task_name classification \
#   --is_training 1 \
#   --root_path ./dataset/EthanolConcentration/ \
#   --model_id EthanolConcentration_STFT \
#   --model UTSClassification \
#   --data UEA \
#   --batch_size 32 \
#   --learning_rate 0.0001 \
#   --train_epochs 30 \
#   --time2img STFT \
#   --lradj fixed \
#   --backbone vit_base \
#   --resize default \
#   --patience 15 \


# echo "  ########################################################"
# echo "  #    ######      ########    ##     ##    ########     #"
# echo "  #    ##     ##   ##     ##   ###    ##    ##           #"
# echo "  #    ##      ##  ##     ##   ## ##  ##    ##           #"
# echo "  #    ##       ## ##     ##   ##  ## ##    ######       #"
# echo "  #    ##       ## ##     ##   ##   ####    ##           #"
# echo "  #    ##      ##  ##     ##   ##    ###    ##           #"
# echo "  #    ##     ##   ##     ##   ##     ##    ##           #"
# echo "  #    ######      ########    ##     ##    ########     #"
# echo "  ########################################################"

# CUDA_VISIBLE_DEVICES=1 python -u run.py \
#   --task_name classification \
#   --is_training 1 \
#   --root_path ./dataset/JapaneseVowels/ \
#   --model_id JapaneseVowels_STFT \
#   --model UTSClassification \
#   --data UEA \
#   --batch_size 32 \
#   --learning_rate 0.0001 \
#   --train_epochs 30 \
#   --time2img STFT \
#   --backbone vit_base \
#   --resize default \
#   --patience 15

# echo "  ########################################################"
# echo "  #    ######      ########    ##     ##    ########     #"
# echo "  #    ##     ##   ##     ##   ###    ##    ##           #"
# echo "  #    ##      ##  ##     ##   ## ##  ##    ##           #"
# echo "  #    ##       ## ##     ##   ##  ## ##    ######       #"
# echo "  #    ##       ## ##     ##   ##   ####    ##           #"
# echo "  #    ##      ##  ##     ##   ##    ###    ##           #"
# echo "  #    ##     ##   ##     ##   ##     ##    ##           #"
# echo "  #    ######      ########    ##     ##    ########     #"
# echo "  ########################################################"



# CUDA_VISIBLE_DEVICES=1 python -u run.py \
#   --task_name classification \
#   --is_training 1 \
#   --root_path ./dataset/Handwriting/ \
#   --model_id Handwriting_STFT \
#   --model UTSClassification \
#   --data UEA \
#   --batch_size 32 \
#   --learning_rate 0.00001 \
#   --train_epochs 30 \
#   --time2img STFT \
#   --lradj fixed \
#   --backbone vit_base \
#   --resize default \
#   --patience 15

# echo "  ########################################################"
# echo "  #    ######      ########    ##     ##    ########     #"
# echo "  #    ##     ##   ##     ##   ###    ##    ##           #"
# echo "  #    ##      ##  ##     ##   ## ##  ##    ##           #"
# echo "  #    ##       ## ##     ##   ##  ## ##    ######       #"
# echo "  #    ##       ## ##     ##   ##   ####    ##           #"
# echo "  #    ##      ##  ##     ##   ##    ###    ##           #"
# echo "  #    ##     ##   ##     ##   ##     ##    ##           #"
# echo "  #    ######      ########    ##     ##    ########     #"
# echo "  ########################################################"


# CUDA_VISIBLE_DEVICES=1 python -u run.py \
#   --task_name classification \
#   --is_training 1 \
#   --root_path ./dataset/Heartbeat/ \
#   --model_id Heartbeat_STFT \
#   --model UTSClassification \
#   --data UEA \
#   --batch_size 4 \
#   --learning_rate 0.0001 \
#   --train_epochs 30 \
#   --time2img STFT \
#   --backbone vit_base \
#   --resize default \
#   --patience 15

# echo "  ########################################################"
# echo "  #    ######      ########    ##     ##    ########     #"
# echo "  #    ##     ##   ##     ##   ###    ##    ##           #"
# echo "  #    ##      ##  ##     ##   ## ##  ##    ##           #"
# echo "  #    ##       ## ##     ##   ##  ## ##    ######       #"
# echo "  #    ##       ## ##     ##   ##   ####    ##           #"
# echo "  #    ##      ##  ##     ##   ##    ###    ##           #"
# echo "  #    ##     ##   ##     ##   ##     ##    ##           #"
# echo "  #    ######      ########    ##     ##    ########     #"
# echo "  ########################################################"

# CUDA_VISIBLE_DEVICES=1 python -u run.py \
#   --task_name classification \
#   --is_training 1 \
#   --root_path ./dataset/SelfRegulationSCP1/ \
#   --model_id SelfRegulationSCP1_STFT \
#   --model UTSClassification \
#   --data UEA \
#   --batch_size 32 \
#   --learning_rate 0.0001 \
#   --lradj fixed \
#   --train_epochs 30 \
#   --time2img STFT \
#   --backbone vit_base \
#   --resize default \
#   --patience 15


# echo "  ########################################################"
# echo "  #    ######      ########    ##     ##    ########     #"
# echo "  #    ##     ##   ##     ##   ###    ##    ##           #"
# echo "  #    ##      ##  ##     ##   ## ##  ##    ##           #"
# echo "  #    ##       ## ##     ##   ##  ## ##    ######       #"
# echo "  #    ##       ## ##     ##   ##   ####    ##           #"
# echo "  #    ##      ##  ##     ##   ##    ###    ##           #"
# echo "  #    ##     ##   ##     ##   ##     ##    ##           #"
# echo "  #    ######      ########    ##     ##    ########     #"
# echo "  ########################################################"


# CUDA_VISIBLE_DEVICES=1 python -u run.py \
#   --task_name classification \
#   --is_training 1 \
#   --root_path ./dataset/SelfRegulationSCP2/ \
#   --model_id SelfRegulationSCP2_STFT \
#   --model UTSClassification \
#   --data UEA \
#   --batch_size 32 \
#   --learning_rate 0.0001 \
#   --train_epochs 30 \
#   --time2img STFT \
#   --backbone vit_base \
#   --resize default \
#   --patience 15

# echo "  ########################################################"
# echo "  #    ######      ########    ##     ##    ########     #"
# echo "  #    ##     ##   ##     ##   ###    ##    ##           #"
# echo "  #    ##      ##  ##     ##   ## ##  ##    ##           #"
# echo "  #    ##       ## ##     ##   ##  ## ##    ######       #"
# echo "  #    ##       ## ##     ##   ##   ####    ##           #"
# echo "  #    ##      ##  ##     ##   ##    ###    ##           #"
# echo "  #    ##     ##   ##     ##   ##     ##    ##           #"
# echo "  #    ######      ########    ##     ##    ########     #"
# echo "  ########################################################"


CUDA_VISIBLE_DEVICES=1 python -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path ./dataset/SpokenArabicDigits/ \
  --model_id SpokenArabicDigits_STFT \
  --model UTSClassification \
  --data UEA \
  --batch_size 16 \
  --learning_rate 0.0001 \
  --train_epochs 30 \
  --time2img STFT \
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


# CUDA_VISIBLE_DEVICES=1 python -u run.py \
#   --task_name classification \
#   --is_training 1 \
#   --root_path ./dataset/UWaveGestureLibrary/ \
#   --model_id UWaveGestureLibrary_STFT \
#   --model UTSClassification \
#   --data UEA \
#   --batch_size 32 \
#   --learning_rate 0.0001 \
#   --train_epochs 30 \
#   --time2img STFT \
#   --backbone vit_base \
#   --resize default \
#   --patience 15

# echo "  ########################################################"
# echo "  #    ######      ########    ##     ##    ########     #"
# echo "  #    ##     ##   ##     ##   ###    ##    ##           #"
# echo "  #    ##      ##  ##     ##   ## ##  ##    ##           #"
# echo "  #    ##       ## ##     ##   ##  ## ##    ######       #"
# echo "  #    ##       ## ##     ##   ##   ####    ##           #"
# echo "  #    ##      ##  ##     ##   ##    ###    ##           #"
# echo "  #    ##     ##   ##     ##   ##     ##    ##           #"
# echo "  #    ######      ########    ##     ##    ########     #"
# echo "  ########################################################"