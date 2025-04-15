import torch
from torch.utils.data import DataLoader
from transformers import T5Tokenizer
from config import Paths, device, TEST_SPLIT, BATCH_SIZE, SUBSET_SEQUENCE
from data_loader import load_and_preprocess_data
from dataset import CodeCPGDataset, collate_fn
from model import CodeCPGSummarizer
from train import train_model
from utils import setup_environment
import os
import sys

def create_dataloaders(data_df, tokenizer):
    """Create train and test dataloaders from DataFrame"""
    test_size = int(len(data_df) * TEST_SPLIT)
    train_df = data_df.iloc[:-test_size]
    test_df = data_df.iloc[-test_size:]
    
    train_loader = DataLoader(
        CodeCPGDataset(train_df, tokenizer),
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4
    )
    
    test_loader = DataLoader(
        CodeCPGDataset(test_df, tokenizer, is_test=True),
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn, 
        num_workers=4
    )
    
    return train_loader, test_loader

def main():
    try:
        setup_environment()
        print(f"Using device: {device}")

        # Initialize single model and tokenizer
        tokenizer = T5Tokenizer.from_pretrained("t5-base")
        model = CodeCPGSummarizer().to(device)
        os.makedirs("models", exist_ok=True)

        # Cumulative training across subsets
        for subset in SUBSET_SEQUENCE:
            print(f"\n=== Training on Subset {subset} ===")
            paths = Paths(subset=subset)
            
            try:
                # Load subset data
                data = load_and_preprocess_data(paths)
                if len(data) == 0:
                    print(f"Warning: Subset {subset} is empty, skipping")
                    continue
                    
                print(f"Loaded {len(data)} samples from subset {subset}")
                
                # Create dataloaders
                train_loader, test_loader = create_dataloaders(data, tokenizer)
                
                # Train on current subset (updates the same model)
                model, metrics = train_model(model, train_loader, test_loader, tokenizer)
                
                # Save intermediate checkpoint
                torch.save(model.state_dict(), f"models/subset_{subset}_checkpoint.pt")
                print(f"Subset {subset} training complete. Metrics:")
                print(f"Loss: {metrics['train_losses']:.4f}, Accuracy: {metrics['train_accuracies']:.4f}")
                
            except Exception as e:
                print(f"Error processing subset {subset}: {str(e)}")
                continue

        # Save final cumulative model
        torch.save(model.state_dict(), "models/final_model.pt")
        print("\nTraining completed successfully!")
        print("Final model saved to models/final_model.pt")
        
        return model
        
    except Exception as e:
        print(f"Fatal error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    trained_model = main()