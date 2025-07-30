from datasets import load_dataset
import torch
import os

# Tải dataset từ Hugging Face
try:
    ds = load_dataset("ml1996/webqsp")
except Exception as e:
    print(f"Không thể tải dataset 'ml1996/webqsp'. Lỗi: {e}")
    print("Vui lòng kiểm tra tên dataset hoặc sử dụng nguồn khác.")
    exit(1)

# Giả sử dataset có split 'train' và 'test'
train_data = ds['train'] if 'train' in ds else None
test_data = ds['test'] if 'test' in ds else None

if train_data is None or test_data is None:
    print("Dataset không chứa split 'train' hoặc 'test'. Vui lòng kiểm tra cấu trúc.")
    exit(1)

# Chuyển đổi thành PyTorch Dataset
def convert_to_pytorch(dataset_split):
    data = []
    entity_ids = [f'Q{i}' for i in range(101)]  # Giả sử 101 thực thể
    embeddings = torch.randn(101, 768)  # Tạo embeddings ngẫu nhiên cho các thực thể
    for item in dataset_split:
        question = item.get('question', '')
        if not question:  # Bỏ qua nếu question rỗng
            continue
        answers = item.get('answers', item.get('answer', ['unknown']))  # Giá trị mặc định nếu rỗng
        if not answers:
            answers = ['unknown']
        input_ids = torch.tensor([hash(word) % 10000 for word in question.split()[:10]], dtype=torch.long).unsqueeze(0)  # Thêm chiều batch
        attention_mask = torch.ones_like(input_ids)  # Tạo mask tương ứng
        labels = torch.tensor([hash(ans) % 10000 for ans in answers[:1]], dtype=torch.long)  # Không unsqueeze, thành (batch_size,)
        if labels.numel() == 0:
            print(f"Warning: Empty labels for question: {question}, answers: {answers}")
            continue
        data.append((input_ids, attention_mask, labels))
    return data, embeddings

train_pytorch, train_embeddings = convert_to_pytorch(train_data)
test_pytorch, test_embeddings = convert_to_pytorch(test_data)

# Lưu thành file .pt
os.makedirs("data/webqsp", exist_ok=True)
torch.save(train_pytorch, "data/webqsp/train.pt")
torch.save(test_pytorch, "data/webqsp/test.pt")
torch.save(train_embeddings, "data/webqsp/embeddings.pt")
print("Đã tạo train.pt, test.pt và embeddings.pt thành công!")