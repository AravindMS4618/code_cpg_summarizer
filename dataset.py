import torch
from torch.utils.data import Dataset
from transformers import T5Tokenizer
import json
from config import MAX_CODE_LENGTH, MAX_SUMMARY_LENGTH, NUM_LABELS, LABEL_MAPPING

class CodeCPGDataset(Dataset):
    def __init__(self, data, tokenizer, max_code_length=MAX_CODE_LENGTH, 
                 max_summary_length=MAX_SUMMARY_LENGTH, is_test=False):
        self.data = data
        self.tokenizer = tokenizer
        self.max_code_length = max_code_length
        self.max_summary_length = max_summary_length
        self.is_test = is_test

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        code_text = item['code']
        code_tokens = item['code_tokens']
        summary = item['docstring']

        # Parse CPG from JSON string if it's stored as a string
        if isinstance(item['cpg'], str):
            cpg = json.loads(item['cpg'])
        else:
            cpg = item['cpg']

        # Use pre-tokenized code tokens if available
        if isinstance(code_tokens, list) and code_tokens:
            # Convert tokens to IDs
            code_input_ids = self.tokenizer.convert_tokens_to_ids(code_tokens)

            # Truncate if needed
            if len(code_input_ids) > self.max_code_length - 2:  # Account for special tokens
                code_input_ids = code_input_ids[:self.max_code_length - 2]

            # Add special tokens
            code_input_ids = [self.tokenizer.pad_token_id] + code_input_ids + [self.tokenizer.eos_token_id]

            # Pad to max_length
            padding_length = self.max_code_length - len(code_input_ids)
            code_input_ids.extend([self.tokenizer.pad_token_id] * padding_length)

            # Create attention mask
            code_attention_mask = [1] * (self.max_code_length - padding_length) + [0] * padding_length

            # Convert to tensors
            code_input_ids = torch.tensor(code_input_ids)
            code_attention_mask = torch.tensor(code_attention_mask)
        else:
            # Fallback to direct tokenization if code_tokens is not available
            code_encoding = self.tokenizer(
                code_text,
                max_length=self.max_code_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            code_input_ids = code_encoding.input_ids.squeeze(0)
            code_attention_mask = code_encoding.attention_mask.squeeze(0)

        # Tokenize summary
        summary_encoding = self.tokenizer(
            summary,
            max_length=self.max_summary_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        # Process CPG
        vertices = cpg.get("vertices", [])
        edges = cpg.get("edges", [])

        # Create one-hot encoded node features
        node_features = torch.zeros((len(vertices), NUM_LABELS))
        for i, vertex in enumerate(vertices):
            label = vertex.get("label", "OTHER")
            label_idx = LABEL_MAPPING.get(label, LABEL_MAPPING["OTHER"])
            node_features[i, label_idx] = 1.0

        # Create edge index for GAT
        if len(edges) > 0:
            edge_index = torch.tensor([[e["outV"], e["inV"]] for e in edges], dtype=torch.long).t()
        else:
            # Handle empty graphs
            edge_index = torch.zeros((2, 0), dtype=torch.long)

        # For test set, also return raw code and docstring for evaluation
        if self.is_test:
            return {
                "code_input_ids": code_input_ids,
                "code_attention_mask": code_attention_mask,
                "node_features": node_features,
                "edge_index": edge_index,
                "num_nodes": len(vertices),
                "summary_input_ids": summary_encoding.input_ids.squeeze(0),
                "summary_attention_mask": summary_encoding.attention_mask.squeeze(0),
                "raw_code": code_text,
                "raw_summary": summary,
                "raw_cpg": cpg
            }
        else:
            return {
                "code_input_ids": code_input_ids,
                "code_attention_mask": code_attention_mask,
                "node_features": node_features,
                "edge_index": edge_index,
                "num_nodes": len(vertices),
                "summary_input_ids": summary_encoding.input_ids.squeeze(0),
                "summary_attention_mask": summary_encoding.attention_mask.squeeze(0),
            }

def collate_fn(batch):
    max_nodes = max(item["num_nodes"] for item in batch)

    # Initialize tensors
    code_input_ids = torch.stack([item["code_input_ids"] for item in batch])
    code_attention_mask = torch.stack([item["code_attention_mask"] for item in batch])
    summary_input_ids = torch.stack([item["summary_input_ids"] for item in batch])
    summary_attention_mask = torch.stack([item["summary_attention_mask"] for item in batch])

    # Prepare node features with padding
    node_features_list = []
    edge_indices_list = []
    batch_indices = []

    node_offset = 0
    for i, item in enumerate(batch):
        num_nodes = item["num_nodes"]

        # Pad node features
        padded_features = torch.zeros((max_nodes, NUM_LABELS))
        padded_features[:num_nodes] = item["node_features"]
        node_features_list.append(padded_features)

        # Adjust edge indices and add to list
        if num_nodes > 0 and item["edge_index"].size(1) > 0:
            edge_index = item["edge_index"]
            edge_indices_list.append(edge_index + node_offset)

            # Create batch indices for nodes
            batch_indices.extend([i] * num_nodes)

            node_offset += num_nodes

    # Concatenate tensors
    node_features_batch = torch.stack(node_features_list)

    # Handle empty graphs if any
    if edge_indices_list:
        edge_index_batch = torch.cat(edge_indices_list, dim=1)
    else:
        edge_index_batch = torch.zeros((2, 0), dtype=torch.long)

    batch_indices_tensor = torch.tensor(batch_indices, dtype=torch.long) if batch_indices else torch.zeros(0, dtype=torch.long)

    # Create a mask for nodes (1 for real nodes, 0 for padding)
    node_mask = torch.zeros((len(batch), max_nodes), dtype=torch.bool)
    for i, item in enumerate(batch):
        node_mask[i, :item["num_nodes"]] = True

    result = {
        "code_input_ids": code_input_ids,
        "code_attention_mask": code_attention_mask,
        "node_features": node_features_batch,
        "edge_index": edge_index_batch,
        "batch_indices": batch_indices_tensor,
        "node_mask": node_mask,
        "summary_input_ids": summary_input_ids,
        "summary_attention_mask": summary_attention_mask,
    }

    # For test set, also include raw texts for evaluation
    if "raw_code" in batch[0]:
        result["raw_code"] = [item["raw_code"] for item in batch]
        result["raw_summary"] = [item["raw_summary"] for item in batch]
        result["raw_cpg"] = [item["raw_cpg"] for item in batch]

    return result