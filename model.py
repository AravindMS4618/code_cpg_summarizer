import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch_geometric.nn import GATConv
from transformers import T5ForConditionalGeneration, T5Tokenizer
from transformers.modeling_outputs import BaseModelOutput
from config import D_MODEL, NUM_HEADS, NUM_LABELS, device

class GraphEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, heads=2):
        super(GraphEncoder, self).__init__()
        self.gat1 = GATConv(input_dim, hidden_dim, heads=heads)
        self.gat2 = GATConv(hidden_dim * heads, output_dim, heads=1)
        self.projection = nn.Linear(output_dim, D_MODEL)

    def forward(self, x, edge_index, batch_indices=None):
        # Handle empty graphs
        if edge_index.size(1) == 0 or x.size(0) == 0:
            return torch.zeros(x.size(0), D_MODEL, device=x.device)

        x = F.elu(self.gat1(x, edge_index))
        x = self.gat2(x, edge_index)
        x = self.projection(x)  # Project to d_model dimension
        return x

class CrossAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(CrossAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        # Query, Key, Value projections for code->graph attention
        self.c2g_query = nn.Linear(d_model, d_model)
        self.c2g_key = nn.Linear(d_model, d_model)
        self.c2g_value = nn.Linear(d_model, d_model)

        # Query, Key, Value projections for graph->code attention
        self.g2c_query = nn.Linear(d_model, d_model)
        self.g2c_key = nn.Linear(d_model, d_model)
        self.g2c_value = nn.Linear(d_model, d_model)

        # Output projections
        self.c2g_out = nn.Linear(d_model, d_model)
        self.g2c_out = nn.Linear(d_model, d_model)

        # Layer normalization
        self.code_norm1 = nn.LayerNorm(d_model)
        self.code_norm2 = nn.LayerNorm(d_model)
        self.graph_norm1 = nn.LayerNorm(d_model)
        self.graph_norm2 = nn.LayerNorm(d_model)

        # Feed-forward networks
        self.code_ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Linear(d_model * 4, d_model)
        )

        self.graph_ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Linear(d_model * 4, d_model)
        )

    def forward(self, code_emb, graph_emb, code_mask=None, graph_mask=None):
        batch_size = code_emb.size(0)
        seq_len = code_emb.size(1)
        graph_len = graph_emb.size(1)

        # Check if graph embeddings are valid (not empty)
        if graph_len == 0:
            return code_emb, graph_emb

        # Code -> Graph attention
        q_c = self.c2g_query(code_emb).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k_g = self.c2g_key(graph_emb).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v_g = self.c2g_value(graph_emb).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Graph -> Code attention
        q_g = self.g2c_query(graph_emb).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k_c = self.g2c_key(code_emb).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v_c = self.g2c_value(code_emb).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Apply scaled dot-product attention: Code -> Graph
        scores_c2g = torch.matmul(q_c, k_g.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Apply masks if provided
        if graph_mask is not None:
            graph_mask = graph_mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, G]
            scores_c2g = scores_c2g.masked_fill(~graph_mask, float('-inf'))

        attn_c2g = F.softmax(scores_c2g, dim=-1)
        out_c2g = torch.matmul(attn_c2g, v_g)  # [B, H, C, D]
        out_c2g = out_c2g.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        out_c2g = self.c2g_out(out_c2g)

        # Apply scaled dot-product attention: Graph -> Code
        scores_g2c = torch.matmul(q_g, k_c.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Apply masks if provided
        if code_mask is not None:
            code_mask = code_mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, C]
            scores_g2c = scores_g2c.masked_fill(~code_mask, float('-inf'))

        attn_g2c = F.softmax(scores_g2c, dim=-1)
        out_g2c = torch.matmul(attn_g2c, v_c)  # [B, H, G, D]
        out_g2c = out_g2c.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        out_g2c = self.g2c_out(out_g2c)

        # Apply residual connections and layer normalization for code
        code_residual = code_emb + out_c2g
        code_norm = self.code_norm1(code_residual)
        code_ffn_out = self.code_ffn(code_norm)
        code_out = self.code_norm2(code_norm + code_ffn_out)

        # Apply residual connections and layer normalization for graph
        graph_residual = graph_emb + out_g2c
        graph_norm = self.graph_norm1(graph_residual)
        graph_ffn_out = self.graph_ffn(graph_norm)
        graph_out = self.graph_norm2(graph_norm + graph_ffn_out)

        return code_out, graph_out

class CodeCPGSummarizer(nn.Module):
    def __init__(self, t5_model_name="t5-base"):
        super(CodeCPGSummarizer, self).__init__()

        # Load T5 model and tokenizer
        self.t5_model = T5ForConditionalGeneration.from_pretrained(t5_model_name)
        self.tokenizer = T5Tokenizer.from_pretrained(t5_model_name)

        # Extract T5 encoder and decoder
        self.code_encoder = self.t5_model.encoder
        self.decoder = self.t5_model.decoder
        self.lm_head = self.t5_model.lm_head

        # Graph encoder
        self.graph_encoder = GraphEncoder(
            input_dim=NUM_LABELS,
            hidden_dim=64,
            output_dim=64
        )

        # Cross-attention between code and graph
        self.cross_attention = CrossAttention(
            d_model=D_MODEL,
            num_heads=NUM_HEADS
        )

    def encode(self, code_input_ids, code_attention_mask, node_features, edge_index, node_mask=None):
        # Encode code text
        code_emb = self.code_encoder(
            input_ids=code_input_ids,
            attention_mask=code_attention_mask
        ).last_hidden_state

        batch_size = code_input_ids.size(0)

        # Check if node_features is empty
        if node_features.size(1) == 0:
            return code_emb

        # Process graph structures batch by batch
        graphs_emb_list = []
        valid_batch_items = 0

        for i in range(batch_size):
            # Extract graph features for this batch item
            batch_node_mask = node_mask[i] if node_mask is not None else None
            num_nodes = batch_node_mask.sum().item() if batch_node_mask is not None else node_features[i].size(0)

            # Skip if no nodes
            if num_nodes == 0:
                graphs_emb_list.append(torch.zeros((1, D_MODEL), device=code_emb.device))
                continue

            valid_batch_items += 1
            # Filter out padding nodes
            nodes_i = node_features[i, :num_nodes]

            # Find relevant edges for this batch
            if edge_index.size(1) > 0:
                batch_edge_mask = (edge_index[0] >= 0) & (edge_index[0] < num_nodes) & (edge_index[1] >= 0) & (edge_index[1] < num_nodes)
                local_edges = edge_index[:, batch_edge_mask]

                # If no edges found, create a self-loop
                if local_edges.size(1) == 0:
                    local_edges = torch.tensor([[0], [0]], device=edge_index.device) if num_nodes > 0 else torch.zeros((2, 0), device=edge_index.device)
            else:
                local_edges = torch.tensor([[0], [0]], device=edge_index.device) if num_nodes > 0 else torch.zeros((2, 0), device=edge_index.device)

            # Encode graph
            graph_emb = self.graph_encoder(nodes_i, local_edges)
            graphs_emb_list.append(graph_emb)

        if valid_batch_items == 0:
            return code_emb

        # Create padded graph embeddings tensor
        max_nodes = max(emb.size(0) for emb in graphs_emb_list) if graphs_emb_list else 1
        padded_graphs_emb = torch.zeros((batch_size, max_nodes, D_MODEL), device=code_emb.device)

        for i, emb in enumerate(graphs_emb_list):
            if emb.size(0) > 0:
                padded_graphs_emb[i, :emb.size(0)] = emb

        # Apply cross-attention between code and graph
        enhanced_code_emb, _ = self.cross_attention(
            code_emb,
            padded_graphs_emb,
            code_attention_mask.bool(),
            node_mask
        )

        return enhanced_code_emb

    def forward(
        self,
        code_input_ids,
        code_attention_mask,
        node_features,
        edge_index,
        node_mask,
        summary_input_ids=None,
        summary_attention_mask=None,
        labels=None
    ):
        # Encode inputs
        encoder_outputs = self.encode(
            code_input_ids,
            code_attention_mask,
            node_features,
            edge_index,
            node_mask
        )
        
        # Training mode
        if summary_input_ids is not None:
            decoder_input_ids = self.shift_tokens_right(summary_input_ids)
            
            decoder_outputs = self.decoder(
                input_ids=decoder_input_ids,
                attention_mask=summary_attention_mask,
                encoder_hidden_states=encoder_outputs.last_hidden_state if hasattr(encoder_outputs, 'last_hidden_state') else encoder_outputs,
                encoder_attention_mask=code_attention_mask
            )
            
            lm_logits = self.lm_head(decoder_outputs[0])
            
            if labels is not None:
                loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
                shifted_labels = summary_input_ids.clone()
                shifted_labels[:, :-1] = summary_input_ids[:, 1:]
                shifted_labels[:, -1] = self.tokenizer.pad_token_id
                
                loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), shifted_labels.view(-1))
                
                # Calculate accuracy
                preds = torch.argmax(lm_logits, dim=-1)
                valid_mask = (shifted_labels != self.tokenizer.pad_token_id) & (shifted_labels != -100)
                correct = (preds == shifted_labels) & valid_mask
                accuracy = correct.sum().float() / valid_mask.sum().float() if valid_mask.sum() > 0 else torch.tensor(0.0)

                return {
                    "loss": loss,
                    "logits": lm_logits,
                    "accuracy": accuracy,
                    "encoder_outputs": encoder_outputs
                }
            
            return {"logits": lm_logits, "encoder_outputs": encoder_outputs}
        
        # Evaluation mode
        return encoder_outputs

    def shift_tokens_right(self, input_ids):
        decoder_start_token_id = self.t5_model.config.decoder_start_token_id
        pad_token_id = self.tokenizer.pad_token_id

        shifted_input_ids = input_ids.new_zeros(input_ids.shape)
        shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
        shifted_input_ids[:, 0] = decoder_start_token_id

        # Replace pad tokens
        shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

        return shifted_input_ids

    def generate(self, code_text=None, cpg_graph=None, code_input_ids=None, code_attention_mask=None, max_length=128, beam_size=4):
        """Generate summary from either raw inputs or tokenized inputs"""
        # If tokenized inputs are provided, use them directly
        if code_input_ids is not None and code_attention_mask is not None:
            code_tokens = {
                "input_ids": code_input_ids.to(device),
                "attention_mask": code_attention_mask.to(device)
            }
        # Otherwise, tokenize from raw text
        elif code_text is not None:
            code_tokens = self.tokenizer(
                code_text,
                max_length=MAX_CODE_LENGTH,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            ).to(device)
        else:
            raise ValueError("Either code_text or (code_input_ids, code_attention_mask) must be provided")

        # Process CPG - Add this conversion to handle string JSON
        if isinstance(cpg_graph, str):
            cpg_graph = json.loads(cpg_graph)

        # Now access the vertices and edges
        vertices = cpg_graph.get("vertices", [])
        edges = cpg_graph.get("edges", [])

        # Handle empty graph
        if not vertices:
            # Just encode the code without graph information
            encoder_hidden_states = self.code_encoder(
                input_ids=code_tokens["input_ids"],
                attention_mask=code_tokens["attention_mask"]
            ).last_hidden_state
            encoder_outputs = self.get_encoder_outputs(encoder_outputs)
        else:
            # Create one-hot encoded node features
            node_features = torch.zeros((1, len(vertices), NUM_LABELS), device=device)
            for i, vertex in enumerate(vertices):
                label = vertex.get("label", "OTHER")
                label_idx = LABEL_MAPPING.get(label, LABEL_MAPPING["OTHER"])
                node_features[0, i, label_idx] = 1.0

            # Create edge index
            if len(edges) > 0:
                edge_index = torch.tensor([[e["outV"], e["inV"]] for e in edges], dtype=torch.long).t().to(device)
            else:
                edge_index = torch.zeros((2, 0), dtype=torch.long, device=device)

            # Create node mask
            node_mask = torch.ones((1, len(vertices)), dtype=torch.bool, device=device)

            # Encode
            encoder_hidden_states = self.encode(
                code_tokens["input_ids"],
                code_tokens["attention_mask"],
                node_features,
                edge_index,
                node_mask
            )
            encoder_outputs = self.get_encoder_outputs(encoder_outputs)

        # Generate with beam search
        beam_output = self.t5_model.generate(
            input_ids=None,
            encoder_outputs=BaseModelOutput(last_hidden_state=encoder_hidden_states),
            attention_mask=code_tokens["attention_mask"],
            max_length=max_length,
            num_beams=beam_size,
            early_stopping=True
        )

        # Decode the generated token IDs
        summary = self.tokenizer.decode(beam_output[0], skip_special_tokens=True)

        return summary
    
    def get_encoder_outputs(self, outputs):
        if isinstance(outputs, BaseModelOutput):
            return outputs
        elif isinstance(outputs, dict):
            return BaseModelOutput(last_hidden_state=outputs.get("encoder_outputs"))
        return BaseModelOutput(last_hidden_state=outputs)