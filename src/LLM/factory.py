from src.LLM.LLM import LLM
from transformers import AutoModel, AutoTokenizer
import torch

def llm_factory(embedding_llm_type, embedding_llm_name, batch_size, device, bits=None):
    """Tạo instance của mô hình ngôn ngữ dựa trên loại và tên."""
    if embedding_llm_type.lower() == "gpt-2":
        class GPT2LLM(LLM):
            def __init__(self, model_name_or_path, device):
                super().__init__(model_name_or_path, device)
                self.batch_size = batch_size

            def embed(self, texts):
                """Tạo embedding cho một danh sách các đoạn text."""
                inputs = self._tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self._device)
                outputs = self._model(**inputs)
                return outputs.last_hidden_state.mean(dim=1)

        return GPT2LLM(embedding_llm_name, device)
    else:
        raise ValueError(f"Unsupported LLM type: {embedding_llm_type}")

    # Nếu bits (quantization) được cung cấp, có thể thêm logic tối ưu hóa sau
    if bits is not None:
        print(f"Quantization with {bits} bits is not implemented yet.")