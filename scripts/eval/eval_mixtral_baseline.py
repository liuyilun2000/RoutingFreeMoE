import argparse
import torch
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, MixtralForCausalLM, MixtralConfig
import torch.nn.functional as F


def evaluate_model(
    model_dir: str,
    max_samples: int = None,
    batch_size: int = 8,
    max_length: int = 512,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    """
    Evaluate a Mixtral model on TinyStories test set.
    
    Metrics:
    - Perplexity
    - Token-level accuracy (next token prediction)
    """
    print(f"Loading model from {model_dir}...")
    
    # Load model and tokenizer
    config = MixtralConfig.from_pretrained(model_dir)
    model = MixtralForCausalLM.from_pretrained(
        model_dir,
        config=config,
        torch_dtype=torch.bfloat16,
        device_map=device,
    )
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'  # For batch generation
    
    print(f"Model loaded. Device: {device}")
    print(f"Model config: {config.num_hidden_layers} layers, {config.num_local_experts} experts")
    
    # Load TinyStories validation set
    print("Loading TinyStories dataset...")
    dataset = load_dataset(
        "parquet",
        data_files={
            "validation": "hf://datasets/roneneldan/TinyStories/data/validation-*.parquet",
        },
    )
    test_dataset = dataset["validation"]
    
    if max_samples is not None:
        test_dataset = test_dataset.select(range(min(max_samples, len(test_dataset))))
    
    print(f"Evaluating on {len(test_dataset)} samples...")
    
    # Tokenize function
    def tokenize(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors="pt",
        )
    
    # Metrics
    total_loss = 0.0
    total_tokens = 0
    total_correct = 0
    total_predictions = 0
    
    # Process in batches
    num_batches = (len(test_dataset) + batch_size - 1) // batch_size
    
    with torch.no_grad():
        for i in tqdm(range(0, len(test_dataset), batch_size), desc="Evaluating"):
            batch_texts = test_dataset[i:i+batch_size]["text"]
            
            # Tokenize batch
            encodings = tokenizer(
                batch_texts,
                truncation=True,
                max_length=max_length,
                padding="max_length",
                return_tensors="pt",
            )
            
            input_ids = encodings["input_ids"].to(device)
            attention_mask = encodings["attention_mask"].to(device)
            
            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids,  # For language modeling, labels = input_ids
            )
            
            loss = outputs.loss
            logits = outputs.logits
            
            # Calculate metrics
            # Shift for next token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()
            shift_mask = attention_mask[..., 1:].contiguous()
            
            # Token-level accuracy
            predictions = shift_logits.argmax(dim=-1)
            correct = (predictions == shift_labels) & (shift_mask == 1)
            total_correct += correct.sum().item()
            total_predictions += shift_mask.sum().item()
            
            # Perplexity calculation
            # Calculate loss only on non-padded tokens
            loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
            token_losses = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            ).view(shift_labels.size())
            
            masked_losses = token_losses * shift_mask
            total_loss += masked_losses.sum().item()
            total_tokens += shift_mask.sum().item()
    
    # Calculate final metrics
    avg_loss = total_loss / total_tokens
    perplexity = np.exp(avg_loss)
    accuracy = total_correct / total_predictions * 100
    
    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    print(f"Model: {model_dir}")
    print(f"Test samples: {len(test_dataset)}")
    print(f"Total tokens evaluated: {total_tokens}")
    print("-" * 50)
    print(f"Average Loss: {avg_loss:.4f}")
    print(f"Perplexity: {perplexity:.4f}")
    print(f"Token Accuracy: {accuracy:.2f}%")
    print("=" * 50)
    
    return {
        "loss": avg_loss,
        "perplexity": perplexity,
        "accuracy": accuracy,
        "total_tokens": total_tokens,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate Mixtral model on TinyStories")
    parser.add_argument(
        "--model-dir",
        type=str,
        default="/home/hk-project-p0022189/hgf_mxv5488/RoutingFreeMoE/output_baseline_mixtral/1_mixtral_baseline_12L_128Dx12E_top2/final_model",
        help="Path to the model directory",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to evaluate (None for all)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for evaluation",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Maximum sequence length",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for evaluation",
    )
    
    args = parser.parse_args()
    
    results = evaluate_model(
        model_dir=args.model_dir,
        max_samples=args.max_samples,
        batch_size=args.batch_size,
        max_length=args.max_length,
        device=args.device,
    )
    
    return results


if __name__ == "__main__":
    main()
