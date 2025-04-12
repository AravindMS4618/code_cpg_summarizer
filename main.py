import torch
from torch.utils.data import DataLoader
from transformers import T5Tokenizer
from config import Paths, device, TEST_SPLIT, BATCH_SIZE, SUBSET_SEQUENCE
from data_loader import load_and_preprocess_data
from dataset import CodeCPGDataset, collate_fn
from model import CodeCPGSummarizer
from train import train_model, evaluate_model
from utils import setup_environment
import sys

def train_on_subsets():
    """Train model sequentially on subsets of data"""
    try:
        setup_environment()
        print(f"Using device: {device}")

        # Initialize model and tokenizer once
        tokenizer = T5Tokenizer.from_pretrained("t5-base")
        model = CodeCPGSummarizer().to(device)
        
        # Training loop through subsets
        for subset in SUBSET_SEQUENCE:
            print(f"\n=== Training on Subset {subset} ===")
            paths = Paths(subset)
            
            # Load subset data
            try:
                data_df = load_and_preprocess_data(paths)
                print(f"Loaded {len(data_df)} samples from subset {subset}")
            except Exception as e:
                print(f"Error loading subset {subset}: {e}")
                continue

            # Split data
            test_size = int(len(data_df) * TEST_SPLIT)
            train_df = data_df.iloc[:-test_size]
            test_df = data_df.iloc[-test_size:]

            # Create dataloaders
            train_dataset = CodeCPGDataset(train_df, tokenizer)
            test_dataset = CodeCPGDataset(test_df, tokenizer, is_test=True)
            
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

            # Train on current subset
            model, _ = train_model(model, train_loader, test_loader, tokenizer)
            
            # Save intermediate checkpoint
            torch.save(model.state_dict(), f"models/subset_{subset}_checkpoint.pt")
            print(f"Saved checkpoint for subset {subset}")
        
        # Final evaluation on full dataset
        print("\n=== Final Evaluation on Full Dataset ===")
        full_paths = Paths()  # Initialize without subset
        try:
            full_data = load_and_preprocess_data(full_paths)
            print(f"Loaded {len(full_data)} samples for final evaluation")
            
            # Create full test dataset
            test_size = int(len(full_data) * TEST_SPLIT)
            test_df = full_data.iloc[-test_size:]
            test_dataset = CodeCPGDataset(test_df, tokenizer, is_test=True)
            
            test_loader = DataLoader(
                test_dataset,
                batch_size=BATCH_SIZE,
                shuffle=False,
                collate_fn=collate_fn,
                num_workers=4
            )
            
            final_metrics = evaluate_model(model, test_loader, tokenizer)
            
            # Save final model
            torch.save(model.state_dict(), "models/final_model.pt")
            
            print("\nFinal Results:")
            print(f"BLEU: {final_metrics['bleu']:.4f}")
            print(f"ROUGE-L F1: {final_metrics['rouge-l']['f']:.4f}")
            print(f"METEOR: {final_metrics['meteor']:.4f}")
            
            return model, final_metrics
            
        except Exception as e:
            print(f"Error during final evaluation: {e}")
            return None

    except Exception as e:
        print(f"Fatal error during training: {e}")
        return None

if __name__ == "__main__":
    result = train_on_subsets()
    if result is None:
        print("\nTraining failed. Check error messages above.")
        sys.exit(1)
    trained_model, metrics = result
    print("\nTraining completed successfully!")