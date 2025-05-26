from PIL import Image
import os

def pil_loader(path: str) -> Image.Image:
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")

def test_image_loader(img_path):
    try:
        img = pil_loader(img_path)
        print(f"✅ 이미지 로딩 성공: {img_path}")
        print(f"크기: {img.size}, 모드: {img.mode}")
        img.save("test_output.jpg")  # 확인용으로 파일로 저장
        print("📝 'test_output.jpg'로 저장했습니다. 확인해보세요.")
    except Exception as e:
        print(f"❌ 이미지 로딩 실패: {img_path}")
        print(f"에러 메시지: {e}")


# ⛳ 여기에 이미지 경로 입력
test_image_loader("/hdd/disc21/R160648.jpg")