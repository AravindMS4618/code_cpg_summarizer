import torch
import time
from datetime import timedelta
from tqdm import tqdm
from torch.optim import AdamW
import nltk
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from rouge import Rouge
from nltk.translate.meteor_score import meteor_score
from config import device, LEARNING_RATE, NUM_EPOCHS

def calculate_metrics(references, hypotheses):
    # Prepare for BLEU
    tokenized_refs = [[nltk.word_tokenize(ref)] for ref in references]
    tokenized_hyps = [nltk.word_tokenize(hyp) for hyp in hypotheses]

    # Calculate BLEU
    smoothie = SmoothingFunction().method1
    bleu_score = corpus_bleu(tokenized_refs, tokenized_hyps, smoothing_function=smoothie)

    # Calculate ROUGE
    rouge = Rouge()
    rouge_scores = rouge.get_scores(hypotheses, references, avg=True)

    # Calculate METEOR
    meteor_scores = [meteor_score([ref], hyp) for ref, hyp in zip(references, hypotheses)]
    avg_meteor = sum(meteor_scores) / len(meteor_scores) if meteor_scores else 0

    return {
        'bleu': bleu_score,
        'rouge-1': rouge_scores['rouge-1'],
        'rouge-2': rouge_scores['rouge-2'],
        'rouge-l': rouge_scores['rouge-l'],
        'meteor': avg_meteor
    }

def train_model(model, train_loader, test_loader, tokenizer, num_epochs=NUM_EPOCHS, lr=LEARNING_RATE):
    # Set up optimizer
    optimizer = AdamW(model.parameters(), lr=lr)

    # Track best model
    best_bleu = 0.0
    best_model_state = None

    # Track training progress
    train_losses = []
    train_accuracies = []
    test_metrics = []

    # Start time
    start_time = time.time()

    # Clear GPU cache before training
    torch.cuda.empty_cache()

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_accuracy = 0.0
        num_batches = 0

        # Training loop
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch in progress_bar:
            # Move batch to device
            code_input_ids = batch["code_input_ids"].to(device)
            code_attention_mask = batch["code_attention_mask"].to(device)
            node_features = batch["node_features"].to(device)
            edge_index = batch["edge_index"].to(device)
            node_mask = batch["node_mask"].to(device)
            summary_input_ids = batch["summary_input_ids"].to(device)
            summary_attention_mask = batch["summary_attention_mask"].to(device)

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(
                code_input_ids=code_input_ids,
                code_attention_mask=code_attention_mask,
                node_features=node_features,
                edge_index=edge_index,
                node_mask=node_mask,
                summary_input_ids=summary_input_ids,
                summary_attention_mask=summary_attention_mask,
                labels=summary_input_ids
            )

            loss = outputs["loss"]
            accuracy = outputs["accuracy"]

            # Backward pass
            loss.backward()

            # Clip gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Update weights
            optimizer.step()

            # Track progress
            epoch_loss += loss.item()
            epoch_accuracy += accuracy.item()
            num_batches += 1

            # Update progress bar
            progress_bar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "accuracy": f"{accuracy.item():.4f}"
            })

        # Calculate epoch averages
        avg_loss = epoch_loss / num_batches if num_batches > 0 else 0
        avg_accuracy = epoch_accuracy / num_batches if num_batches > 0 else 0

        # Save for plotting
        train_losses.append(avg_loss)
        train_accuracies.append(avg_accuracy)

        # Evaluate on test set
        test_results = evaluate_model(model, test_loader, tokenizer)
        test_metrics.append(test_results)

        # Print epoch summary
        elapsed_time = time.time() - start_time
        print(f"Epoch {epoch+1}/{num_epochs} completed in {timedelta(seconds=int(elapsed_time))}")
        print(f"Train Loss: {avg_loss:.4f}, Train Accuracy: {avg_accuracy:.4f}")
        print(f"Test BLEU: {test_results['bleu']:.4f}, ROUGE-L: {test_results['rouge-l']['f']:.4f}, METEOR: {test_results['meteor']:.4f}")

        # Save best model
        if test_results['bleu'] > best_bleu:
            best_bleu = test_results['bleu']
            best_model_state = model.state_dict().copy()
            print(f"New best model with BLEU score: {best_bleu:.4f}")

        print("-" * 50)

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # Clear GPU cache after training
    torch.cuda.empty_cache()

    return model, {"train_losses": train_losses, "train_accuracies": train_accuracies, "test_metrics": test_metrics}

def evaluate_model(model, test_loader, tokenizer):
    model.eval()
    references = []
    hypotheses = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            # Move batch to device
            code_input_ids = batch["code_input_ids"].to(device)
            code_attention_mask = batch["code_attention_mask"].to(device)
            node_features = batch["node_features"].to(device)
            edge_index = batch["edge_index"].to(device)
            node_mask = batch["node_mask"].to(device)

            # Get reference summaries
            raw_summaries = batch["raw_summary"]
            references.extend(raw_summaries)

            # Generate predictions
            encoder_outputs = model.encode(
                code_input_ids=code_input_ids,
                code_attention_mask=code_attention_mask,
                node_features=node_features,
                edge_index=edge_index,
                node_mask=node_mask
            )

            # Generate summaries
            generated_ids = model.t5_model.generate(
                input_ids=None,
                encoder_outputs=BaseModelOutput(last_hidden_state=encoder_outputs),
                attention_mask=code_attention_mask,
                max_length=MAX_SUMMARY_LENGTH,
                num_beams=4,
                early_stopping=True
            )

            # Decode generated summaries
            generated_summaries = [
                tokenizer.decode(g, skip_special_tokens=True) for g in generated_ids
            ]
            hypotheses.extend(generated_summaries)

    # Calculate metrics
    metrics = calculate_metrics(references, hypotheses)

    return metrics