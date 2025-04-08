import torch
from torch.utils.data import DataLoader, random_split
from config import Paths, device, TEST_SPLIT, BATCH_SIZE
from data_loader import load_and_preprocess_data
from dataset import CodeCPGDataset, collate_fn
from model import CodeCPGSummarizer
from train import train_model, evaluate_model
from utils import setup_environment

def main():
    # Setup environment
    setup_environment()
    
    # Initialize paths
    paths = Paths()
    
    print(f"Using device: {device}")

    # Load and preprocess data
    print("Loading data...")
    try:
        data_df = load_and_preprocess_data(paths)
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # Initialize tokenizer
    tokenizer = T5Tokenizer.from_pretrained("t5-base")

    # Split data into train and test sets
    test_size = int(len(data_df) * TEST_SPLIT)
    train_df = data_df.iloc[:-test_size]
    test_df = data_df.iloc[-test_size:]

    print(f"Train set size: {len(train_df)}")
    print(f"Test set size: {len(test_df)}")

    # Create datasets
    train_dataset = CodeCPGDataset(train_df, tokenizer)
    test_dataset = CodeCPGDataset(test_df, tokenizer, is_test=True)

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4
    )

    # Initialize model
    print("Initializing model...")
    model = CodeCPGSummarizer().to(device)

    # Train model
    print("Starting training...")
    model, training_history = train_model(model, train_loader, test_loader, tokenizer)

    # Save model
    print("Saving model...")
    torch.save(model.state_dict(), paths.model_save_path)

    # Final evaluation
    print("Final evaluation...")
    final_metrics = evaluate_model(model, test_loader, tokenizer)

    print("\nFinal Results:")
    print(f"BLEU: {final_metrics['bleu']:.4f}")
    print(f"ROUGE-1 F1: {final_metrics['rouge-1']['f']:.4f}")
    print(f"ROUGE-2 F1: {final_metrics['rouge-2']['f']:.4f}")
    print(f"ROUGE-L F1: {final_metrics['rouge-l']['f']:.4f}")
    print(f"METEOR: {final_metrics['meteor']:.4f}")

    return model, final_metrics, training_history

if __name__ == "__main__":
    trained_model, metrics, history = main()