import pandas as pd
from pathlib import Path

EXCEL_PATH = Path(r"c:\Users\laiweicheng\Downloads\Taiwanese\Taiwanese\Image_info.xls")

print("=== Image_info.xls 結構分析 ===\n")

# 讀取不指定 header，查看前幾行
df_raw = pd.read_excel(EXCEL_PATH, header=None)
print("全表大小:", df_raw.shape)
print("\n前 10 行 (無 header):")
print(df_raw.head(10))

print("\n" + "="*50)
print("每列含義推測：")
print("="*50)

# 找 header 行
try:
    header_row_idx = df_raw[df_raw.iloc[:, 0] == 'file_name'].index[0]
    print(f"\nHeader 行 (第 {header_row_idx} 行):")
    header = df_raw.iloc[header_row_idx]
    print(header)
    
    print("\n資料行範例 (header 後的前 5 行):")
    data_rows = df_raw.iloc[header_row_idx+1:header_row_idx+6]
    for i, (idx, row) in enumerate(data_rows.iterrows()):
        print(f"\n樣本 {i+1}:")
        for col_idx, val in enumerate(row):
            print(f"  欄 {col_idx} ({header.iloc[col_idx]}): {val}")
            
except Exception as e:
    print(f"錯誤: {e}")

print("\n" + "="*50)
print("統計資訊：")
print("="*50)
df = df_raw.iloc[header_row_idx+1:].copy()
df.columns = df_raw.iloc[header_row_idx]

print(f"總樣本數: {len(df)}")
print(f"欄數: {len(df.columns)}")
print(f"\n欄名稱:")
for i, col in enumerate(df.columns):
    print(f"  {i}: {col}")
