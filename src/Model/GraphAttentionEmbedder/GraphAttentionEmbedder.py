import torch
import torch.nn as nn

class GraphAttentionEmbedder(nn.Module):
    def __init__(self, config, llm):
        super(GraphAttentionEmbedder, self).__init__()
        self.config = config
        self.llm = llm
        self.embedding_dim = llm.embedding_length  # 768 từ GPT-2
        self.attention = nn.Linear(self.embedding_dim * 3, self.embedding_dim)
        self.dropout = nn.Dropout(p=0.1)  # Thêm dropout để tránh overfitting

    @classmethod
    def from_config(cls, config, llm):
        """Khởi tạo mô hình từ cấu hình."""
        return cls(config, llm)

    def forward(self, central_node_embedding, node_embeddings, edge_embeddings):
        """Thực hiện forward pass với attention mechanism."""
        combined = torch.cat([central_node_embedding, node_embeddings, edge_embeddings], dim=-1)
        attention_output = self.attention(combined)
        return self.dropout(attention_output)  # Áp dụng dropout

    def load_state_dict(self, state_dict):
        """Tải trạng thái mô hình từ file."""
        super().load_state_dict(state_dict)

    def save(self, path):
        """Lưu mô hình vào file."""
        torch.save(self.state_dict(), path)

    def evaluate(self, central_node_embedding, node_embeddings, edge_embeddings, labels):
        """Đánh giá mô hình (giả định có labels để tính loss)."""
        outputs = self.forward(central_node_embedding, node_embeddings, edge_embeddings)
        loss_fn = nn.MSELoss()  # Giả định loss function, có thể thay đổi
        loss = loss_fn(outputs, labels)
        return loss