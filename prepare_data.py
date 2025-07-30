from datasets import load_dataset
import os
import json  # Thêm dòng này để nhập module json

# Stream dataset wiki40b với ngôn ngữ Ả Rập (thay 'ar' bằng 'en' nếu muốn tiếng Anh)
ds = load_dataset("google/wiki40b", "en", split="train", streaming=True)

# Tạo thư mục trex_json_files nếu chưa có
os.makedirs("trex_json_files", exist_ok=True)

# Lấy 100 mẫu đầu tiên từ stream và tạo file .json
sample_count = 0
for example in ds.take(100):  # Lấy 100 mẫu để thử
    # Tạo triple mẫu (dựa trên ID Wikidata và giả định predicate/object)
    sample_data = {
        "subject": example["wikidata_id"],  # ID Wikidata từ bài viết
        "predicate": "P31",  # Ví dụ: instance of
        "object": "Q5"       # Ví dụ: human (có thể thay bằng giá trị thực tế)
    }
    with open(f"trex_json_files/data_{sample_count}.json", "w", encoding="utf-8") as f:
        json.dump(sample_data, f)
    sample_count += 1

print(f"Đã tạo {sample_count} tệp .json trong trex_json_files.")