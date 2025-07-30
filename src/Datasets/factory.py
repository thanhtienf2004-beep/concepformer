from typing import Dict
import networkx as nx
from tqdm import tqdm
from src.Datasets.TRExStar import TRExStar
import json

def trex_star_factory(dataset_name):
    """
    Factory method to create a TRExStar dataset instance and prepare it.
    """
    if dataset_name in ["TRExStar", "trex_json_files"]:
        trex_star_builder = TRExStar()
        trex_star_builder.download_and_prepare()
        # Trả về DatasetDict
        dataset = trex_star_builder.as_dataset()
        print(f"Debug: trex_star_factory returned dataset with splits: {dataset.keys()} from file {__file__}")
        return dataset
    raise ValueError(f"Dataset {dataset_name} not supported")

def trex_star_graphs_factory(dataset_name):
    """
    Factory method to create a dictionary of networkx graphs from a TRExStar dataset.
    """
    dataset = trex_star_factory(dataset_name)
    graphs = {}
    # Chọn split "train" và lấy độ dài
    train_dataset = dataset['train']
    print(f"Debug: Processing train dataset with length {len(train_dataset)} from file {__file__}")
    for index in tqdm(range(len(train_dataset)), desc="Loading nx graphs"):
        datapoint = train_dataset[index]  # Truy cập bằng chỉ số sau khi chọn split
        # datapoint là dict chứa 'entity' và 'json'
        entity = datapoint['entity']
        json_data = datapoint['json']
        print(f"Debug in factory: entity {entity}, json_data: {json_data}, type: {type(json_data)} from file {__file__}")  # Debug trong factory
        # Đảm bảo json_data là dict
        if isinstance(json_data, str):
            try:
                json_data = json.loads(json_data)
            except json.JSONDecodeError as e:
                raise ValueError(f"Failed to decode json_data for entity {entity}: {e}")
        if not isinstance(json_data, dict):
            raise ValueError(f"json_data for entity {entity} is not a dict: {json_data}")
        # Chuyển đổi cấu trúc nếu cần
        if "nodes" in json_data and isinstance(json_data["nodes"], dict) and "id" in json_data["nodes"]:
            json_data["nodes"] = [{"id": id} for id in json_data["nodes"]["id"]]
        if "links" in json_data and isinstance(json_data["links"], dict) and all(key in json_data["links"] for key in ["source", "target", "predicate"]):
            json_data["links"] = [{"source": s, "target": t, "predicate": p} for s, t, p in zip(json_data["links"]["source"], json_data["links"]["target"], json_data["links"]["predicate"])]
        # Kiểm tra cấu trúc nested
        if "nodes" not in json_data or "links" not in json_data:
            raise ValueError(f"Missing 'nodes' or 'links' key in json_data for entity {entity}: {json_data}")
        if not isinstance(json_data["nodes"], list) or not all("id" in node for node in json_data["nodes"]):
            raise ValueError(f"Invalid nodes structure in json_data for entity {entity}: {json_data}")
        if not isinstance(json_data["links"], list) or not all("source" in link and "target" in link and "predicate" in link for link in json_data["links"]):
            raise ValueError(f"Invalid links structure in json_data for entity {entity}: {json_data}")
        graphs[entity] = nx.node_link_graph(json_data, edges="links")
    return graphs