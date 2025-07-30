from src.GraphAligner.BigGraphAligner import BigGraphAligner
from src.LLM.LLM import LLM
import torch
import re
import os

def main(config, graphs, device):
    llm = LLM()
    # Tạo embeddings ban đầu
    graph_aligner = BigGraphAligner(llm, graphs, config.graph_dataset_name, use_untrained=True)
    graph_aligner.prepare()
    graph_aligner.build_index()

    # Khởi tạo và huấn luyện baseline (không dùng KG)
    baseline_model = BigGraphAligner(llm, graphs, config.graph_dataset_name, use_untrained=False, use_kg=False, epochs=20)
    optimizer_base = torch.optim.Adam(baseline_model.parameters(), lr=0.001)
    train_data = None  # Khởi tạo train_data với giá trị mặc định
    # Giả sử train_data từ WebQSP hoặc T-Rex Star
    try:
        train_data = torch.load("data/webqsp/train.pt")  # Cần chuyển đổi trước
        baseline_model.train(train_data, optimizer_base, device)
    except FileNotFoundError:
        print("File train.pt không tồn tại. Bỏ qua huấn luyện baseline.")
    # Tạo thư mục model_checkpoint nếu không tồn tại
    os.makedirs("model_checkpoint", exist_ok=True)
    torch.save(baseline_model.state_dict(), "model_checkpoint/baseline_model.pth")

    # Khởi tạo và huấn luyện ConceptFormer (dùng KG)
    conceptformer_model = BigGraphAligner(llm, graphs, config.graph_dataset_name, use_untrained=False, use_kg=True, epochs=20)
    optimizer_cf = torch.optim.Adam(conceptformer_model.parameters(), lr=0.01)
    embeddings = graph_aligner.node_embedding_batch(list(graph_aligner.entity_index.keys()))
    if train_data is not None:
        conceptformer_model.train(train_data, optimizer_cf, device, embeddings)
    # Tạo thư mục model_checkpoint nếu không tồn tại
    os.makedirs("model_checkpoint", exist_ok=True)
    torch.save(conceptformer_model.state_dict(), "model_checkpoint/conceptformer_trained.pth")

    # Đánh giá (cần test data)
    try:
        test_data = torch.load("data/webqsp/test.pt")  # Cần chuyển đổi trước
        def evaluate(model, kg_embeds=None):
            model.eval()
            total, correct = 0, 0
            with torch.no_grad():
                for batch in test_data:
                    input_ids, attention_mask, labels = batch[:3]  # Giả sử định dạng
                    outputs = model(input_ids.to(device), attention_mask.to(device), kg_embeds.to(device) if kg_embeds is not None else None)
                    preds = torch.argmax(outputs, dim=-1)
                    correct += (preds == labels.to(device)).sum().item()
                    total += labels.size(0)
            return correct / total if total > 0 else 0.0

        base_acc = evaluate(baseline_model)
        cf_embeds = graph_aligner.node_embedding_batch(test_data[0][3]) if test_data[0][3] else None
        cf_acc = evaluate(conceptformer_model, cf_embeds)
        print(f"Accuracy Baseline: {base_acc:.4f}")
        print(f"Accuracy ConceptFormer: {cf_acc:.4f}")
    except FileNotFoundError:
        print("File test.pt không tồn tại. Bỏ qua đánh giá.")

    # Hỏi-đáp
    def answer_question(model, graph_aligner, llm, question):
        entities = re.findall(r'Q\d+', question)
        embeddings = graph_aligner.node_embedding_batch(entities) if entities and model.use_kg else None
        input_ids = llm.tokenize(question)
        attention_mask = torch.ones_like(input_ids)
        with torch.no_grad():
            outputs = model(input_ids.to(device), attention_mask.to(device), embeddings.to(device) if embeddings is not None else None)
            return llm.late_generate_from_logits(outputs)

    while True:
        question = input("Vui lòng nhập câu hỏi (nhập 'quit' để thoát): ")
        if question.lower() == 'quit':
            break
        base_answer = answer_question(baseline_model, graph_aligner, llm, question)
        cf_answer = answer_question(conceptformer_model, graph_aligner, llm, question)
        print("Câu hỏi:", question)
        print("Trả lời Baseline:", base_answer)
        print("Trả lời ConceptFormer:", cf_answer)

if __name__ == "__main__":
    device = torch.device("cpu")
    config = type('Config', (), {'graph_dataset_name': 'TRExStar'})()  # Placeholder
    graphs = {}  # Giả sử đã tải từ factory.py
    main(config, graphs, device)