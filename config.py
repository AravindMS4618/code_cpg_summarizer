import os
import torch

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Constants
MAX_CODE_LENGTH = 1024
MAX_SUMMARY_LENGTH = 256
BATCH_SIZE = 4
LEARNING_RATE = 2e-4
NUM_EPOCHS = 1
D_MODEL = 768  # T5 hidden dimension
NUM_HEADS = 8  # For cross-attention
TEST_SPLIT = 0.2  # 20% of data for testing

# Label mapping for CPG nodes
LABEL_MAPPING = {
    "TYPE_DECL": 0, "METHOD_DECL": 1, "PARAM": 2, "VAR_DECL": 3,
    "BLOCK": 4, "RETURN": 5, "IF": 6, "FOR": 7, "WHILE": 8, "CALL": 9,
    "ASSIGN": 10, "FIELD_ACCESS": 11, "BINARY_OP": 12, "UNARY_OP": 13,
    "LITERAL": 14, "EXPR": 15, "OTHER": 16
}
NUM_LABELS = len(LABEL_MAPPING)

# Path configurations
class Paths:
    def __init__(self, subset=None):
        self.data_root = "data"
        self.current_subset = subset
        self.code_dir = os.path.join(self.data_root, f"subset_{subset}/code") if subset else None
        self.cpg_dir = os.path.join(self.data_root, f"subset_{subset}/cpg") if subset else None
        self.model_save_path = "models/code_cpg_summarizer_model.pt"
        
        os.makedirs("models", exist_ok=True)

SUBSET_SEQUENCE = [1, 2, 3, 4]  # Ordered list of subsets to train on

# Set environment variables for memory optimization
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"