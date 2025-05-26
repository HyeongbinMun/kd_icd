import torch

# 파일 경로
pt_path = "/workspace/results/1000_mobilenetv2/predictions.pt"

# 로드 (GPU 저장된 경우도 CPU에서 읽도록 처리)
data = torch.load(pt_path, map_location='cpu')

print(f"\n✅ 로드 완료: {type(data)}")

# 딕셔너리인 경우
if isinstance(data, dict):
    print(f"📌 딕셔너리 항목 수: {len(data)}\n")
    for i, (key, value) in enumerate(data.items()):
        print(f"[{i}] 🔑 Key: {key}")
        print(f"    Type: {type(value)}")

        if isinstance(value, list):
            print(f"    Length: {len(value)}")
            print(f"    Sample values:")
            for j, item in enumerate(value[:3]):
                print(f"      [{j}] {item}")

        elif isinstance(value, dict):
            print(f"    Dict keys: {list(value.keys())[:3]}")
            print(f"    Sample values:")
            for k in list(value.keys())[:3]:
                print(f"      {k}: {value[k]}")

        elif isinstance(value, torch.Tensor):
            print(f"    Shape: {value.shape}")
            print(f"    Dtype: {value.dtype}")
            print(f"    Length: {value.numel()}")
            print(f"    Sample values: {value[:10].tolist()}")  # 앞 10개 값만 출력

        else:
            print(f"    Value: {value}")

        if i == 4:  # 필요에 따라 출력 키 개수 조정
            break

else:
    print("❗ 예상과 다르게 dict가 아닙니다.")
