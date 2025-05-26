import torch

# 경로 설정
pt_path = "/workspace/results/test/predictions.pt"

# .pt 파일 로딩
predictions = torch.load(pt_path)

print(f"✅ 타입: {type(predictions)} / 항목 수: {len(predictions)}")

# 딕셔너리인 경우 키와 값 샘플 출력
if isinstance(predictions, dict):
    print("\n📌 Top-level keys:")
    for i, key in enumerate(predictions.keys()):
        print(f"  [{i}] Key: {key}")
        if i == 4:
            break

    # 각 key에 해당하는 value 요약 출력
    for key, value in list(predictions.items())[:3]:
        print(f"\n🔹 Key: {key}")
        print(f"  Type: {type(value)}")
        if isinstance(value, list):
            print(f"  Length: {len(value)}")
            for j, item in enumerate(value[:3]):
                print(f"    [{j}] {item}")
        elif isinstance(value, dict):
            print(f"  Dict keys: {list(value.keys())[:3]}")
        else:
            print(f"  Value: {value}")

    # ✅ 중복 확인 로직 추가 (예: "matches" 키 안에 있는 리스트가 대상일 경우)
    if "matches" in predictions and isinstance(predictions["matches"], list):
        print("\n🔍 중복 (query, db) 쌍 검사 중...")

        matches = predictions["matches"]
        seen = {}
        duplicates = []

        for idx, p in enumerate(matches):
            key = (p.query, p.db)
            if key in seen:
                duplicates.append((key, seen[key], idx))  # (key, first_index, current_index)
            else:
                seen[key] = idx

        if duplicates:
            print(f"❌ 중복된 (query, db) 쌍이 {len(duplicates)}개 있습니다.")
            for i, (key, first_idx, dup_idx) in enumerate(duplicates[:10]):  # 최대 10개 출력
                print(f"  [{i}] 중복 pair: {key} (처음: {first_idx}, 중복: {dup_idx})")
        else:
            print("✅ 중복 없음.")

    else:
        print("\nℹ️ 'matches' 키가 없거나 리스트가 아님. 중복 검사를 건너뜀.")

else:
    print("❗ 예상과 다르게 dict가 아닙니다. 내부 타입 확인이 필요합니다.")
