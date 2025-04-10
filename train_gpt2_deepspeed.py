import torch
import deepspeed
import argparse
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from datasets import load_dataset

def load_small_dataset(tokenizer, seq_len=128, max_samples=1000):
    """Loads and tokenizes a small slice of the Wikipedia dataset."""
    print("Loading a slice of Wikipedia dataset...")
    raw_dataset = load_dataset("nthngdy/oscar-mini", "unshuffled_deduplicated_en",split='train')

    # Use only a subset of samples for quick experimentation
    raw_dataset = raw_dataset.select(range(min(max_samples, len(raw_dataset))))

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=seq_len,
            return_overflowing_tokens=False,
            return_attention_mask=True,
        )

    tokenized_dataset = raw_dataset.map(tokenize_function, batched=True, remove_columns=["text"])

    # Prepare samples for CausalLM with input_ids = labels
    formatted_dataset = []
    for example in tokenized_dataset:
        input_ids = torch.tensor(example["input_ids"])
        attention_mask = torch.tensor(example["attention_mask"])
        formatted_dataset.append({
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": input_ids.clone()
        })

    print(f"Prepared {len(formatted_dataset)} samples from Wikipedia.")
    return formatted_dataset


def main():
    parser = argparse.ArgumentParser(description="DeepSpeed GPT-2 Training Example")
    parser.add_argument('--model_name_or_path', type=str, default='gpt2',
                        help='Hugging Face model name or path (e.g., gpt2, gpt2-medium)')
    parser.add_argument('--epochs', type=int, default=1, help='Number of training epochs')
    parser.add_argument('--local_rank', type=int, default=-1,
                        help='Local rank passed from distributed launcher')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--output_dir', type=str, default='./ds_gpt2_output', help='Output directory for checkpoints')

    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()

    set_seed(args.seed)
    
    print(f"Loading tokenizer: {args.model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print("Added EOS token as PAD token.")
        
    print(f"Loading model: {args.model_name_or_path}")
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)
    model.config.pad_token_id = tokenizer.pad_token_id

    print("Loading and tokenizing Wikipedia dataset...")
    train_dataset = load_small_dataset(tokenizer, seq_len=128, max_samples=500)
    if not train_dataset:
        print("Failed to load dataset. Exiting.")
        return
        
    print("Initializing DeepSpeed...")
    model_engine, optimizer, _, lr_scheduler = deepspeed.initialize(
        args=args,
        model=model,
        model_parameters=model.parameters(),
    )
    
    def collate_fn(batch):
        input_ids = torch.stack([item['input_ids'] for item in batch])
        attention_mask = torch.stack([item['attention_mask'] for item in batch])
        labels = torch.stack([item['labels'] for item in batch])
        return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}

    print(f"Using micro-batch size per GPU: {model_engine.train_micro_batch_size_per_gpu()}")
    print(f"Gradient accumulation steps: {model_engine.gradient_accumulation_steps()}")
    print(f"Effective train batch size: {model_engine.train_batch_size()}")

    print("Starting training...")
    for epoch in range(args.epochs):
        print(f"--- Epoch {epoch+1}/{args.epochs} ---")
        model_engine.train()
        num_samples = len(train_dataset)
        micro_batch_size = model_engine.train_micro_batch_size_per_gpu()
        num_micro_batches = (num_samples + micro_batch_size - 1) // micro_batch_size

        processed_samples = 0
        for i in range(0, num_samples, micro_batch_size):
            batch_list = train_dataset[i : i + micro_batch_size]
            if not batch_list:
                continue

            batch = collate_fn(batch_list)
            device = model_engine.local_rank
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model_engine(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            model_engine.backward(loss)
            model_engine.step()

            processed_samples += len(batch_list)
            if model_engine.local_rank == 0 and (i // micro_batch_size) % 10 == 0:
                print(f"Epoch: {epoch+1}, Step: {(i // micro_batch_size) + 1}/{num_micro_batches}, "
                      f"Processed: {processed_samples}/{num_samples}, "
                      f"Loss: {loss.item():.4f}, LR: {lr_scheduler.get_last_lr()[0]:.2e}")

    if model_engine.local_rank == 0:
        print(f"Saving model checkpoint to {args.output_dir}")
        os.makedirs(args.output_dir, exist_ok=True)
        save_tag = f"final_epoch_{args.epochs}"
        model_engine.save_checkpoint(args.output_dir, save_tag)

    print("Training finished.")

if __name__ == "__main__":
    main()
