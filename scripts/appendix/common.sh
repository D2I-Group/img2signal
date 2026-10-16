export CUDA_VISIBLE_DEVICES=${GPU:-0}

TSC_DATASETS="UWaveGestureLibrary SpokenArabicDigits Handwriting FaceDetection"

TSF_DATASETS=(
  "ETTh1|ETTh1|./dataset/ETT-small/|ETTh1.csv|24|336|7|96 192 336 720"
  "ETTm1|ETTm1|./dataset/ETT-small/|ETTm1.csv|96|336|7|96 192 336 720"
  "Weather|custom|./dataset/weather/|weather.csv|144|336|21|96 192 336 720"
  "Illness|custom|./dataset/illness/|national_illness.csv|52|104|7|24 36 48 60"
)
