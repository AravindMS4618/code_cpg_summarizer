import nltk
import os

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