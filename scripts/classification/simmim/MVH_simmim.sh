# CUDA_VISIBLE_DEVICES=0 python -u run.py \
#   --task_name classification \
#   --is_training 1 \
#   --root_path ./dataset/EthanolConcentration/ \
#   --model_id EthanolConcentration \
#   --model UTSClassification \
#   --data UEA \
#   --batch_size 32 \
#   --learning_rate 0.0001 \
#   --train_epochs 30 \
#   --time2img MVH \
#   --lradj fixed \
#   --backbone simmim \
#   --resize simmim \
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


# CUDA_VISIBLE_DEVICES=0 python -u run.py \
#   --task_name classification \
#   --is_training 1 \
#   --root_path ./dataset/JapaneseVowels/ \
#   --model_id JapaneseVowels \
#   --model UTSClassification \
#   --data UEA \
#   --batch_size 32 \
#   --learning_rate 0.0001 \
#   --train_epochs 30 \
#   --time2img MVH \
#   --backbone simmim \
#   --resize simmim \
#   --lradj fixed \
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

# CUDA_VISIBLE_DEVICES=0 python -u run.py \
#   --task_name classification \
#   --is_training 1 \
#   --root_path ./dataset/Handwriting/ \
#   --model_id Handwriting \
#   --model UTSClassification \
#   --data UEA \
#   --batch_size 128 \
#   --learning_rate 0.0001 \
#   --lradj fixed \
#   --train_epochs 30 \
#   --time2img MVH \
#   --lradj fixed \
#   --backbone simmim \
#   --resize simmim \
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

# CUDA_VISIBLE_DEVICES=0 python -u run.py \
#   --task_name classification \
#   --is_training 1 \
#   --root_path ./dataset/Heartbeat/ \
#   --model_id Heartbeat \
#   --model UTSClassification \
#   --data UEA \
#   --batch_size 4 \
#   --learning_rate 0.0001 \
#   --train_epochs 30 \
#   --time2img MVH \
#   --lradj fixed \
#   --backbone simmim \
#   --resize simmim \
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

# CUDA_VISIBLE_DEVICES=0 python -u run.py \
#   --task_name classification \
#   --is_training 1 \
#   --root_path ./dataset/SelfRegulationSCP1/ \
#   --model_id SelfRegulationSCP1 \
#   --model UTSClassification \
#   --data UEA \
#   --batch_size 32 \
#   --learning_rate 0.0001 \
#   --lradj fixed \
#   --train_epochs 30 \
#   --time2img MVH \
#   --backbone simmim \
#   --resize simmim \
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


# CUDA_VISIBLE_DEVICES=0 python -u run.py \
#   --task_name classification \
#   --is_training 1 \
#   --root_path ./dataset/SelfRegulationSCP2/ \
#   --model_id SelfRegulationSCP2 \
#   --model UTSClassification \
#   --data UEA \
#   --batch_size 32 \
#   --learning_rate 0.0001 \
#   --train_epochs 30 \
#   --time2img MVH \
#   --lradj fixed \
#   --backbone simmim \
#   --resize simmim \
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

# CUDA_VISIBLE_DEVICES=0 python -u run.py \
#   --task_name classification \
#   --is_training 1 \
#   --root_path ./dataset/SpokenArabicDigits/ \
#   --model_id SpokenArabicDigits \
#   --model UTSClassification \
#   --data UEA \
#   --batch_size 16 \
#   --learning_rate 0.0001 \
#   --train_epochs 30 \
#   --time2img MVH \
#   --lradj fixed \
#   --backbone simmim \
#   --resize simmim \
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

# CUDA_VISIBLE_DEVICES=0 python -u run.py \
#   --task_name classification \
#   --is_training 1 \
#   --root_path ./dataset/UWaveGestureLibrary/ \
#   --model_id UWaveGestureLibrary \
#   --model UTSClassification \
#   --data UEA \
#   --batch_size 32 \
#   --learning_rate 0.0001 \
#   --train_epochs 30 \
#   --time2img MVH \
#   --lradj fixed \
#   --backbone simmim \
#   --resize simmim \
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


# CUDA_VISIBLE_DEVICES=0 python -u run.py \
#   --task_name classification \
#   --is_training 1 \
#   --root_path ./dataset/PEMS-SF/ \
#   --model_id PEMS-SF \
#   --model UTSClassification \
#   --data UEA \
#   --batch_size 32 \
#   --learning_rate 0.0001 \
#   --train_epochs 30 \
#   --time2img MVH \
#   --lradj fixed \
#   --backbone simmim \
#   --resize simmim \
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

CUDA_VISIBLE_DEVICES=0 python -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path ./dataset/FaceDetection/ \
  --model_id FaceDetection \
  --model UTSClassification \
  --data UEA \
  --batch_size 32 \
  --learning_rate 0.0001 \
  --train_epochs 30 \
  --time2img MVH \
  --lradj fixed \
  --backbone simmim \
  --resize simmim \
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
