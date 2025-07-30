import os
import csv
import h5py
import torch
import torch.nn as nn
from tqdm import tqdm
import json
import wandb

class BigGraphAligner(nn.Module):
    def __init__(self, llm, graphs, dataset_name, epochs=1000, use_untrained=False, use_kg=True):
        super(BigGraphAligner, self).__init__()
        self.llm = llm
        self.graphs = graphs
        self.dataset_name = dataset_name
        self.epochs = epochs
        self._use_untrained = use_untrained
        self.use_kg = use_kg
        self.entity_index = {}
        self.relation_index = {}
        self.folder = f"data/artifacts/BigGraphAlignment_v{self.epochs}/{self.dataset_name}/{self.llm.name}"
        self.embedding_length = llm.embedding_length if hasattr(llm, 'embedding_length') else 768
        self.num_classes = 10000  # Khớp với num_classes của LLM
        if use_kg:
            self.kg_attention = nn.MultiheadAttention(self.embedding_length, 8)
            self.fc = nn.Linear(self.embedding_length * 2, self.num_classes)  # Thay đổi output thành num_classes
        else:
            self.fc = nn.Linear(self.embedding_length, self.num_classes)  # Thay đổi output thành num_classes

    def prepare(self):
        if os.path.isfile(f'{self.folder}/init/embeddings_entity_0.v0.h5'):
            print("Already prepared")
            return
        os.makedirs(f'{self.folder}/init', exist_ok=True)
        with h5py.File(f'{self.folder}/init/embeddings_entity_0.v0.h5', 'w') as hf:
            embeddings = torch.randn(101, self.embedding_length)  # Tạo embeddings ngẫu nhiên
            hf.create_dataset('embeddings', data=embeddings.numpy())
        with open(f'{self.folder}/entities.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['entity_id', 'entity_name'])  # Header
            for i in range(101):
                writer.writerow([f'Q{i}', f'Entity_{i}'])
        with open(f'{self.folder}/graph_edges.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['source', 'predicate', 'target'])  # Header
            writer.writerow(['Q0', 'P31', 'Q1'])  # Ví dụ
        with open(f'{self.folder}/relations.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['relation_id', 'relation_name'])  # Header
            writer.writerow(['P31', 'instance_of'])  # Ví dụ

    def ensure_relations_file(self):
        relations_path = f'{self.folder}/relations.csv'
        if not os.path.isfile(relations_path):
            with open(relations_path, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['relation_id', 'relation_name'])  # Header
                writer.writerow(['P31', 'instance_of'])  # Ví dụ

    def build_index(self):
        if not os.path.isfile(f'{self.folder}/init/embeddings_entity_0.v0.h5'):
            self.prepare()
        self.ensure_relations_file()  # Đảm bảo relations.csv luôn tồn tại
        with h5py.File(f'{self.folder}/init/embeddings_entity_0.v0.h5', 'r') as hf:
            trained_embeddings = hf['embeddings'][:]
        with open(f'{self.folder}/entities.csv', 'r') as file:
            reader = csv.reader(file)
            next(reader)  # Skip header
            for i, row in tqdm(enumerate(reader), desc='Building entity index', total=trained_embeddings.shape[0]):
                if not row:
                    continue
                try:
                    entity_id, _ = row
                    self.entity_index[entity_id] = torch.from_numpy(trained_embeddings[i, :])
                except ValueError as e:
                    print(f"Error unpacking row {i}: {row}, skipping. Error: {e}")
                    continue
        with open(f'{self.folder}/relations.csv', 'r') as file:
            reader = csv.reader(file)
            next(reader)  # Skip header
            for i, row in tqdm(enumerate(reader), desc='Building relation index'):
                if not row:
                    continue
                try:
                    relation_id, _ = row
                    self.relation_index[relation_id] = i
                except ValueError as e:
                    print(f"Error unpacking row {i}: {row}, skipping. Error: {e}")
                    continue

    def node_embedding(self, entity_id):
        if entity_id in self.entity_index:
            return self.entity_index[entity_id]
        return None

    def node_embedding_batch(self, entity_ids):
        return torch.stack([self.node_embedding(eid) for eid in entity_ids if self.node_embedding(eid) is not None])

    def train(self, train_data, optimizer, device, embeddings=None):
        if self._use_untrained or not self.use_kg:
            print("Using untrained BigGraphAligner or skipping KG training")
            return
        self.training = True  # Đặt trạng thái huấn luyện thủ công
        for epoch in range(self.epochs):
            for batch in train_data:
                input_ids, attention_mask, labels = batch  # Chỉ unpack 3 giá trị
                if labels.numel() == 0:  # Kiểm tra nếu labels rỗng
                    print(f"Warning: Empty labels in batch, skipping. Input_ids: {input_ids}, Labels: {labels}")
                    continue
                kg_embeds = embeddings if embeddings is not None else None  # Sử dụng embeddings đã truyền vào
                optimizer.zero_grad()
                llm_output = self.llm(input_ids.to(device), attention_mask.to(device))  # (batch_size, seq_len, embed_dim)
                if kg_embeds is not None:
                    # Tạo kg_embeds phù hợp với seq_len
                    kg_embeds = kg_embeds.unsqueeze(0).expand(input_ids.size(0), -1, -1)  # (batch_size, num_entities, embed_dim)
                    kg_embeds = kg_embeds[:, :input_ids.size(1), :]  # Cắt để khớp seq_len
                    kg_attended = self.kg_attention(kg_embeds.to(device), kg_embeds.to(device), kg_embeds.to(device))[0]  # (batch_size, seq_len, embed_dim)
                else:
                    kg_attended = None
                combined = torch.cat((llm_output, kg_attended), dim=-1) if kg_attended is not None else llm_output
                # Giảm kích thước outputs về (batch_size, num_classes)
                outputs = self.fc(combined)  # (batch_size, seq_len, embed_dim) -> (batch_size, seq_len, num_classes)
                outputs = outputs[:, -1, :]  # Lấy output của bước cuối cùng (batch_size, num_classes)
                loss = torch.nn.functional.cross_entropy(outputs, labels.to(device))  # labels thành (batch_size,)
                loss.backward()
                optimizer.step()
            print(f"Epoch {epoch+1}/{self.epochs}, Loss: {loss.item():.4f}")
        self.training = False  # Tắt trạng thái huấn luyện sau khi hoàn tất

    def forward(self, input_ids, attention_mask, kg_embeds=None):
        if not self.use_kg or kg_embeds is None:
            return self.llm(input_ids, attention_mask)
        kg_attended, _ = self.kg_attention(kg_embeds, kg_embeds, kg_embeds)
        combined = torch.cat((self.llm(input_ids, attention_mask), kg_attended), dim=-1)
        return self.fc(combined)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)