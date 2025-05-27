python /workspace/sscd/sscd/disc_eval.py \
  --disc_path /hdd/disc21 \
  --gpus=2 \
  --workers=4 \
  --output_path=/workspace/results/1000_mobilenetv2 \
  --size=288 \
  --preserve_aspect_ratio=false \
  --backbone=TV_MOBILENETV2 \
  --dims=320 \
  --model_state=/workspace/weights/mobilenetv2_sscd.pt