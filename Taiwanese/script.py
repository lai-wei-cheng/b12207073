from pathlib import Path
from PIL import Image

# 輸入與輸出資料夾路徑（請自行修改）
input_dir = Path("faces")       # 放原始臉圖的資料夾
output_dir = Path("jpg") # 輸出 256x256 jpg 的資料夾

# 建立輸出資料夾（如果不存在的話）
output_dir.mkdir(parents=True, exist_ok=True)

# 允許的影像格式
valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

for img_path in input_dir.iterdir():
    if img_path.suffix.lower() not in valid_exts:
        continue  # 跳過不是圖片的檔案

    try:
        # 開啟圖片
        img = Image.open(img_path)

        # 轉成 RGB（避免灰階或有 alpha channel 不能存 jpg 的問題）
        img = img.convert("RGB")

        # 直接縮放成 256x256（不保留長寬比例）
        img = img.resize((256, 256), Image.LANCZOS)

        # 輸出檔名：跟原檔同名但副檔名改成 .jpg
        out_path = output_dir / f"{img_path.stem}.jpg"
        img.save(out_path, "JPEG", quality=95)

        print(f"已處理：{img_path.name} -> {out_path.name}")
    except Exception as e:
        print(f"處理 {img_path.name} 時發生錯誤：{e}")

