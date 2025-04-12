import nltk
import os
from torch.utils.data import DataLoader
from dataset import CodeCPGDataset, collate_fn
from config import BATCH_SIZE

def setup_environment():
    # Download necessary NLTK resources
    nltk.download('wordnet')
    nltk.download('omw-1.4')
    nltk.download('punkt')
    nltk.download('punkt_tab')
    
    # Print environment information
    print("Environment setup complete")
    print(f"Working directory: {os.getcwd()}")
    
def generate_docstring(model, code_text, cpg_graph=None):
    """
    Generate a docstring for the given code and optional CPG graph.

    Args:
        model: The trained CodeCPGSummarizer model
        code_text: Source code text as a string
        cpg_graph: Optional CPG graph dict or JSON string

    Returns:
        Generated docstring
    """
    model.eval()

    # Handle CPG if not provided
    if cpg_graph is None:
        # Create an empty CPG graph
        cpg_graph = {"vertices": [], "edges": []}

    # Generate the summary
    with torch.no_grad():
        summary = model.generate(code_text=code_text, cpg_graph=cpg_graph)

    return summary

def split_data(data_df, test_size=0.2):
    test_size = int(len(data_df) * test_size)
    return data_df.iloc[:-test_size], data_df.iloc[-test_size:]

def create_dataloader(df, tokenizer, is_test=False):
    dataset = CodeCPGDataset(df, tokenizer, is_test=is_test)
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=not is_test,
        collate_fn=collate_fn,
        num_workers=4
    )