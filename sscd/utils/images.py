from PIL import Image
import os
from tqdm import tqdm

def check_images_strict(root_dir, extensions={'.jpg', '.jpeg', '.png'}):
    all_files = []
    for root, _, files in os.walk(root_dir):
        for file in files:
            if any(file.lower().endswith(ext) for ext in extensions):
                all_files.append(os.path.join(root, file))

    corrupted = []

    for path in tqdm(all_files, desc="Strict image check (like DataLoader)"):
        try:
            with Image.open(path) as img:
                img = img.convert("RGB")  # torchvision의 default_loader가 내부적으로 수행
                img.load()               # 디코딩까지 수행
        except Exception as e:
            corrupted.append((path, str(e)))

    if corrupted:
        print(f"\n❌ PyTorch-style 로딩 실패 이미지 총 {len(corrupted)}개 발견!")
        with open("corrupted_images_strict.log", "w") as f:
            for path, err in corrupted:
                log_entry = f"{path} ({err})"
                print(f" - {log_entry}")
                f.write(log_entry + "\n")
        print("📝 로그 저장됨: corrupted_images_strict.log")
    else:
        print("\n✅ 모든 이미지가 DataLoader 로딩 기준에서도 정상입니다!")

# 예시 사용
check_images_strict('/hdd/disc21/te')
