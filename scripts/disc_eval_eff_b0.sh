python /workspace/sscd/sscd/disc_eval.py \
  --disc_path /hdd/disc21 \
  --gpus=2 \
  --workers=4 \
  --output_path=/workspace/results/1000_eff_b0 \
  --size 288 \
  --preserve_aspect_ratio false \
  --backbone TV_EFFICIENTNET_B0 \
  --dims 512 \
  --model_state /workspace/weights/eff_b0.pt

