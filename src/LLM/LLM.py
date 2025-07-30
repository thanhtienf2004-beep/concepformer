import torch
import torch.nn as nn

class LLM(nn.Module):
    def __init__(self):
        super(LLM, self).__init__()
        self.embedding_length = 768
        self.name = "llm_model"  # Thuộc tính name
        self.embedding = nn.Embedding(10000, self.embedding_length)  # Giả sử vocab size và embed dim
        self.transformer = nn.Transformer(d_model=self.embedding_length, nhead=8, num_encoder_layers=6, batch_first=True)
        self.fc = nn.Linear(self.embedding_length, 10000)  # Giả sử output vocab size

    def late_embedding(self, indices):
        return self.embedding(indices)

    def tokenize(self, text):
        # Giả sử tokenize đơn giản, trả về tensor 2D (seq_len,) và mở rộng thành batch
        tokens = [hash(word) % 10000 for word in text.split()[:10]]  # Placeholder
        return torch.tensor(tokens, dtype=torch.long).unsqueeze(0)  # Thêm chiều batch (1, seq_len)

    def late_generate_from_logits(self, logits):
        # Giả sử sinh văn bản từ logits
        preds = torch.argmax(logits, dim=-1)
        return " ".join([str(x.item()) for x in preds])  # Placeholder

    def forward(self, input_ids, attention_mask):
        embeddings = self.embedding(input_ids)  # (batch_size, seq_len, embed_dim)
        # Tạo mask attention 2D (seq_len, seq_len) thay vì 3D
        seq_len = input_ids.size(1)
        mask = torch.triu(torch.ones(seq_len, seq_len, device=input_ids.device) * float('-inf'), diagonal=1)
        # Mở rộng mask cho từng head (nhead=8) nếu cần, nhưng PyTorch sẽ tự xử lý với mask 2D
        return self.transformer(src=embeddings, tgt=embeddings, src_mask=mask)