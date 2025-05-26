import torch

# Load checkpoints
ckpt1 = torch.load("/workspace/weights/sscd_disc_mixup.torchvision.pt", map_location="cpu")
ckpt2 = torch.load("/workspace/weights/mobilenetv2.pt", map_location="cpu")

# Extract state_dicts if necessary
sd1 = ckpt1.get("model_state_dict", ckpt1)
sd2 = ckpt2.get("model_state_dict", ckpt2)

# Compare keys
keys1 = set(sd1.keys())
keys2 = set(sd2.keys())

print("❌ Missing in MobileNetV2:")
print(keys1 - keys2)

print("\n⚠️ Extra in MobileNetV2:")
print(keys2 - keys1)