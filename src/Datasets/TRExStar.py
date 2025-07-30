import glob
from typing import List, Dict, Any
from pathlib import Path
import os
import json
import time

from datasets import GeneratorBasedBuilder, SplitGenerator, Split, BuilderConfig, DatasetInfo
from datasets.features import Features, Value, Sequence
import networkx as nx

class TRExStar(GeneratorBasedBuilder):
    VERSION = "1.0.0"
    BUILDER_CONFIGS = [
        BuilderConfig(
            name="TRExStar",
            version=VERSION,
            description="TRExStar includes all relevant sub-graphs from the TREx Dataset. Each subgraph is a star topography with the relevant entity in the center and up to 100 neighbouring entities."
        )
    ]

    def _info(self) -> DatasetInfo:
        return DatasetInfo(
            description="Custom TRExStar dataset from JSON files.",
            features=Features({
                "entity": Value("string"),
                "json": Features({
                    "nodes": Sequence({"id": Value("string")}),
                    "links": Sequence({
                        "source": Value("string"),
                        "target": Value("string"),
                        "predicate": Value("string")
                    })
                })
            }),
            supervised_keys=None,
        )

    def _split_generators(self, dl_manager: Any) -> List[SplitGenerator]:
        trex_json_dir = Path(__file__).parent.parent.parent / 'trex_json_files'
        if not os.path.exists(trex_json_dir):
            raise FileNotFoundError(f"Thư mục {trex_json_dir} không tồn tại. Vui lòng chạy prepare_data.py để tạo file.")
        print(f"Debug: _split_generators executed at {time.ctime()} from file {__file__}")
        return [
            SplitGenerator(
                name=Split.TRAIN,
                gen_kwargs={
                    "trex_star_dir": str(trex_json_dir),
                    "split": "train",
                },
            ),
        ]

    def _generate_examples(self, trex_star_dir: str, split: str) -> Dict[str, Any]:
        graph_files = glob.glob(f"{trex_star_dir}/*.json")
        if not graph_files:
            raise FileNotFoundError(f"Không tìm thấy file .json trong {trex_star_dir}")
        if split != "train":
            raise NotImplementedError(f"Split {split} not implemented.")
        print(f"Debug: _generate_examples started for split {split} at {time.ctime()} from file {__file__}")

        for graph_path in graph_files:
            entity_id = Path(graph_path).stem
            data = {}
            try:
                with open(graph_path, 'r', encoding='utf-8') as f:
                    raw_json = f.read()  # Đọc chuỗi thô
                    triple = json.loads(raw_json)  # Chuyển thành dict
                # Kiểm tra và chuyển đổi triple thành node-link data
                if not all(key in triple for key in ["subject", "predicate", "object"]):
                    raise ValueError(f"File {graph_path} thiếu khóa cần thiết: {triple}")
                node_link_data = {
                    "nodes": [{"id": triple["subject"]}, {"id": triple["object"]}],
                    "links": [{"source": triple["subject"], "target": triple["object"], "predicate": triple["predicate"]}]
                }
                print(f"Debug in TRExStar: entity {entity_id}, raw_json: {raw_json}, triple: {triple}, node_link_data: {node_link_data}, data before yield: {data}")  # Debug chi tiết
                data["json"] = node_link_data  # Gán dict nested
                data["entity"] = entity_id
                print(f"Debug in TRExStar: entity {entity_id}, data after yield: {data}")  # Debug sau khi gán
                # Kiểm tra cấu trúc trước khi yield
                if not isinstance(data["json"], dict) or not isinstance(data["json"]["nodes"], list) or not all("id" in node for node in data["json"]["nodes"]) or not isinstance(data["json"]["links"], list) or not all("source" in link and "target" in link and "predicate" in link for link in data["json"]["links"]):
                    raise ValueError(f"Dữ liệu không đúng cấu trúc trước khi yield cho entity {entity_id}: {data}")
                yield entity_id, data
            except Exception as e:
                print(f"Lỗi đọc file {graph_path}: {e}")
                continue
        print(f"Debug: _generate_examples finished for split {split} at {time.ctime()} from file {__file__}")