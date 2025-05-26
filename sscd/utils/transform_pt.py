import torch

# 기존 mobilenetv2.pt 로드
ckpt = torch.load("/workspace/weights/mobilenetv2.pt", map_location="cpu")

# model_state_dict만 꺼내서 저장
torch.save(ckpt["model_state_dict"], "/workspace/weights/mobilenetv2_sscd.pt")