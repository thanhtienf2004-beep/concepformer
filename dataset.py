from datasets import load_dataset

# Stream dataset wiki40b với ngôn ngữ Ả Rập (hoặc thay 'ar' bằng 'en' nếu muốn tiếng Anh)
ds = load_dataset("google/wiki40b", "en", split="train", streaming=True)