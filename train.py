import torch
import argparse
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
# Ensure Lightning is installed: pip install lightning
from lightning.fabric import Fabric
from lightning.fabric.loggers import TensorBoardLogger # Optional logger
from tqdm import tqdm
import warnings

# Suppress specific warnings if needed (e.g., from Hugging Face)
warnings.filterwarnings("ignore", message=".*but requires Rust dependencies.*")
warnings.filterwarnings("ignore", message=".*TorchScript is not supported.*")


# Define a simple PyTorch Dataset wrapper
class HFDataset(Dataset):
    def __init__(self, data_list):
        self.data = data_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def load_small_dataset(tokenizer, seq_len=128, max_samples=1000):
    """Loads and tokenizes a small slice of a dataset."""
    print(f"Attempting to load and tokenize dataset (max_samples={max_samples}, seq_len={seq_len})...")
    # Using wikitext as it's commonly available and suitable
    
    raw_dataset = load_dataset("nthngdy/oscar-mini", "unshuffled_deduplicated_en",split='train',trust_remote_code=True)

    # Use only a subset of samples for quick experimentation
    if len(raw_dataset) > max_samples:
         print(f"Selecting {max_samples} samples from the dataset.")
         raw_dataset = raw_dataset.select(range(max_samples))
    else:
         print(f"Using all available {len(raw_dataset)} samples.")

    # Filter out potentially empty texts BEFORE tokenization
    original_count = len(raw_dataset)
    raw_dataset = raw_dataset.filter(lambda example: example['text'] and len(example['text'].strip()) > 10) # Filter short/empty
    filtered_count = len(raw_dataset)
    if original_count != filtered_count:
        print(f"Filtered out {original_count - filtered_count} empty or short samples.")

    if filtered_count == 0:
        print("No valid samples found after filtering.")
        return None

    def tokenize_function(examples):
        # Tokenize texts individually first to handle potential errors robustly
        tokenized_outputs = []
        for i, text in enumerate(examples["text"]):
            if not text or len(text.strip()) == 0:
                print(f"Warning: Encountered empty string at index {i} during tokenization batch.")
                # Provide dummy valid output if necessary, or handle upstream filtering better
                output = tokenizer("", padding="max_length", max_length=seq_len, truncation=True, return_attention_mask=True)
            else:
                # Add prefix space for GPT-2/RoBERTa models if needed (often helps)
                # text = " " + text if not text.startswith(" ") else text
                try:
                    output = tokenizer(
                        text,
                        truncation=True,
                        padding="max_length", # Pad to max_length
                        max_length=seq_len,
                        return_attention_mask=True,
                    )
                except Exception as e:
                    print(f"Skipping sample due to tokenization error: {e} | Text: {text[:100]}...")
                    # Append placeholder that matches structure but indicates error or skip
                    output = tokenizer(tokenizer.pad_token * seq_len, padding="max_length", max_length=seq_len, truncation=True, return_attention_mask=True) # Use pad tokens
            tokenized_outputs.append(output)

        # Combine results into the expected batch format
        batch = {}
        if not tokenized_outputs: # Should not happen if filtering is done properly
            return batch
        keys = tokenized_outputs[0].keys()
        for key in keys:
            batch[key] = [output[key] for output in tokenized_outputs]
        return batch


    print("Tokenizing dataset...")
    tokenized_dataset = raw_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"], # Remove original text column
        desc="Tokenizing dataset",
        num_proc=max(os.cpu_count() // 2, 1) # Use multiple processes for tokenization
    )

    # Prepare samples for CausalLM with input_ids = labels
    formatted_dataset = []
    print("Formatting dataset for Causal LM...")
    for example in tokenized_dataset:
        # Ensure keys exist before creating tensors
        if "input_ids" not in example or "attention_mask" not in example:
            print(f"Skipping malformed example: {example}")
            continue
        input_ids = torch.tensor(example["input_ids"])
        attention_mask = torch.tensor(example["attention_mask"])
        formatted_dataset.append({
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": input_ids.clone() # For Causal LM, labels are usually the input_ids
        })

    print(f"Prepared {len(formatted_dataset)} samples.")
    if not formatted_dataset:
         print("Warning: Resulting dataset is empty after processing.")
    return formatted_dataset


def main():
    parser = argparse.ArgumentParser(description="Lightning Fabric GPT-2 FSDP Training Example")
    parser.add_argument('--model_name_or_path', type=str, default='gpt2',
                        help='Hugging Face model name or path (e.g., gpt2, gpt2-medium)')
    parser.add_argument('--epochs', type=int, default=1, help='Number of training epochs')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--output_dir', type=str, default='./fabric_fsdp_gpt2_output', help='Output directory for checkpoints')
    parser.add_argument('--batch_size_per_device', type=int, default=4, help='Batch size per GPU')
    parser.add_argument('--grad_accum_steps', type=int, default=4, help='Gradient accumulation steps')
    parser.add_argument('--learning_rate', type=float, default=5e-5, help='Learning rate')
    parser.add_argument('--warmup_steps', type=int, default=100, help='Linear warmup steps')
    parser.add_argument('--seq_len', type=int, default=128, help='Sequence length for tokenization')
    parser.add_argument('--max_samples', type=int, default=2000, help='Max samples from dataset') # Increased default
    parser.add_argument('--log_interval', type=int, default=10, help='Log every N gradient steps')

    # --- Fabric arguments ---
    # Defaulting strategy to FSDP
    parser.add_argument("--accelerator", type=str, default="auto", help="Accelerator ('cpu', 'cuda', 'mps', 'tpu', 'auto')")
    parser.add_argument("--strategy", type=str, default="fsdp", help="Strategy ('dp', 'ddp', 'fsdp', 'auto') - Set to fsdp")
    parser.add_argument("--devices", type=str, default="auto", help="Devices ('auto', '1', '0,1', '-1' for all)")
    # FSDP often prefers bf16 if available (Ampere+ GPUs)
    parser.add_argument("--precision", type=str, default="bf16-mixed", help="Precision ('32-true', '16-mixed', 'bf16-mixed')")
    parser.add_argument("--num_nodes", type=int, default=1, help="Number of nodes for distributed training")
    # --- End Fabric arguments ---

    args = parser.parse_args()

    # --- Initialize Fabric ---
    # Optional: Add a logger
    logger = TensorBoardLogger(root_dir=os.path.join(args.output_dir, "logs"))
    fabric = Fabric(
        accelerator=args.accelerator,
        strategy=args.strategy, # Explicitly FSDP or from args
        devices=args.devices,
        precision=args.precision,
        num_nodes=args.num_nodes,
        loggers=logger,
    )
    fabric.launch() # Start the distributed processes/setup
    # --- End Fabric Initialization ---

    fabric.seed_everything(args.seed + fabric.global_rank) # Set seed across all processes, add rank for variation

    if fabric.global_rank == 0:
        os.makedirs(args.output_dir, exist_ok=True) # Create output dir only on rank 0

    effective_batch_size = args.batch_size_per_device * fabric.world_size * args.grad_accum_steps
    fabric.print(f"Effective batch size: {effective_batch_size}")
    fabric.print(f"Using Strategy: {type(fabric.strategy).__name__}")
    fabric.print(f"Using Precision: {fabric._precision}")

    fabric.print(f"Loading tokenizer: {args.model_name_or_path}")
    # Load tokenizer once, it's typically small and doesn't need distribution
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        fabric.print("Added EOS token as PAD token.")

    # --- FSDP Aware Model Initialization ---
    # Use init_module for potentially large models with FSDP to avoid OOM on rank 0
    # empty_init=True can save memory by initializing on meta device first
    # Let Fabric decide based on strategy whether empty_init is beneficial
    with fabric.init_module():
        fabric.print(f"Loading model: {args.model_name_or_path} within fabric.init_module()")
        model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)
        model.config.pad_token_id = tokenizer.pad_token_id # Set pad token id
        # Optional: Apply activation checkpointing here if needed for very large models
        # if fabric.strategy.name == "fsdp":
        #     from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
        #     from transformers.models.gpt2.modeling_gpt2 import GPT2Block
        #     auto_wrap_policy = partial(transformer_auto_wrap_policy, transformer_layer_cls={GPT2Block})
        #     # You might need to pass this policy to the FSDPStrategy instance in Fabric initialization
        #     # Or apply it manually if Fabric doesn't expose easy wrapping policy config yet
        #     print("Note: Activation Checkpointing / Auto Wrapping Policy not explicitly applied in this example.")


    fabric.print("Loading and tokenizing dataset...")
    # Load dataset on rank 0 only to avoid redundant downloads/processing
    train_data_list = None
    if fabric.global_rank == 0:
        train_data_list = load_small_dataset(tokenizer, seq_len=args.seq_len, max_samples=args.max_samples)
        if not train_data_list:
            print("FATAL: Failed to load dataset on rank 0. Exiting.")
            # Signal other ranks to exit cleanly if possible (tricky without explicit comms)
    
    # Broadcast dataset list from rank 0 to all ranks
    # Use torch.distributed primitives via Fabric for robust broadcasting
    # Convert to a format suitable for broadcast (e.g., list of tensors might be complex)
    # Simple approach: let all ranks load if dataset is small/cached, or use a shared filesystem
    # More robust: Serialize/deserialize or use Fabric's broadcast - for now, let's assume shared cache/rank 0 loads
    # For simplicity in this example, we assume load_small_dataset is fast enough or cached
    # A more robust solution would involve `fabric.broadcast` for the data list or indices
    if train_data_list is None and fabric.global_rank == 0: # Check if rank 0 failed
         exit(1) # Exit if rank 0 failed
    if fabric.world_size > 1:
         fabric.barrier() # Wait for rank 0 to potentially finish loading/filtering
         # Re-load on other ranks if not using broadcast (relies on cache)
         if fabric.global_rank != 0:
              train_data_list = load_small_dataset(tokenizer, seq_len=args.seq_len, max_samples=args.max_samples)
              if not train_data_list:
                   print(f"FATAL: Failed to load dataset on rank {fabric.global_rank}. Exiting.")
                   exit(1)

    if not train_data_list:
         fabric.print("Dataset is empty or failed to load on all ranks. Exiting.")
         return

    train_dataset = HFDataset(train_data_list)

    def collate_fn(batch):
        keys = batch[0].keys()
        collated_batch = {key: torch.stack([item[key] for item in batch]) for key in keys}
        return collated_batch

    # --- Setup DataLoader (BEFORE setup calls for model/optimizer) ---
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size_per_device,
        shuffle=True, # Shuffling is good practice
        collate_fn=collate_fn,
        num_workers=min(4, os.cpu_count() // fabric.world_size if fabric.world_size > 0 else 1), # Adjust num_workers per process
        pin_memory=True # Can improve transfer speed
    )

    # --- Setup Model FIRST with Fabric (this wraps it with FSDP) ---
    fabric.print("Setting up model with Fabric (applying FSDP)...")
    # --- End Fabric Model Setup ---

    # --- Setup Optimizer and Scheduler AFTER model setup ---
    fabric.print("Setting up optimizer and scheduler...")
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate) # Use model.parameters() AFTER fabric.setup(model)

    # Calculate total training steps
    # We need the length of the dataloader *after* setup for correct calculation in distributed settings
    # Effective steps per epoch = len(fabric_dataloader) // grad_accum_steps
    # Estimate total steps (may be slightly off if dataset size isn't divisible by batch size * world size)
    estimated_steps_per_epoch = len(train_dataset) // (args.batch_size_per_device * fabric.world_size)
    total_training_steps = (estimated_steps_per_epoch // args.grad_accum_steps) * args.epochs
    fabric.print(f"Estimated total training steps: {total_training_steps} (for LR scheduler)")

    if total_training_steps <= 0:
        print("Warning: Calculated total training steps is zero or negative. "
              f"Check dataset size ({len(train_dataset)}), batch size ({args.batch_size_per_device}), "
              f"world size ({fabric.world_size}), and grad accum ({args.grad_accum_steps}). Using warmup steps as total.")
        total_training_steps = args.warmup_steps * 2 # Provide a fallback total

    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=min(args.warmup_steps, total_training_steps), # Ensure warmup <= total
        num_training_steps=total_training_steps
    )
    # --- End Setup Optimizer ---

    # --- Setup Optimizer and Dataloaders with Fabric ---
    fabric.print("Setting up optimizer and dataloaders with Fabric...")
    
    model,optimizer = fabric.setup(model,optimizer)
    #optimizer = fabric.setup(optimizer) # Setup the optimizer
    train_dataloader = fabric.setup_dataloaders(train_dataloader) # Setup dataloader for distributed sampling
    # --- End Fabric Setup ---

    # Calculate steps per epoch accurately now that dataloader is setup
    actual_steps_per_epoch = len(train_dataloader)
    fabric.print(f"Actual steps per epoch per device: {actual_steps_per_epoch}")


    fabric.print("Starting training...")
    total_processed_samples = 0
    current_grad_step = 0 # Track gradient accumulation steps

    for epoch in range(args.epochs):
        fabric.print(f"--- Epoch {epoch+1}/{args.epochs} ---")
        model.train()

        # Use tqdm only on rank 0 for cleaner output
        iterable = tqdm(train_dataloader, desc=f"Epoch {epoch+1}", disable=(fabric.global_rank != 0))

        for batch_idx, batch in enumerate(iterable):
            is_accumulating = (batch_idx + 1) % args.grad_accum_steps != 0

            # Forward pass - FSDP handles sharding/gathering automatically
            # Run under `no_sync` context when accumulating gradients in distributed settings
            # Fabric >= 2.0 handles this automatically within `fabric.backward` based on strategy.
            outputs = model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                labels=batch['labels']
            )
            loss = outputs.loss

            # Normalize loss for gradient accumulation
            loss = loss / args.grad_accum_steps

            # Backward pass - Fabric handles precision scaling and gradient sync internally
            fabric.backward(loss)

            # Step optimizer and scheduler after accumulation steps
            if not is_accumulating:
                optimizer.step()
                optimizer.zero_grad()
                # Step scheduler based on optimizer steps (gradient steps)
                # Avoid stepping if total_training_steps was invalid
                if total_training_steps > 0:
                     lr_scheduler.step()
                current_grad_step += 1

                # Logging (on rank 0)
                if fabric.global_rank == 0 and current_grad_step % args.log_interval == 0:
                    current_lr = lr_scheduler.get_last_lr()[0]
                    # Calculate samples processed in this gradient step across all devices
                    samples_in_step = args.batch_size_per_device * fabric.world_size * args.grad_accum_steps
                    total_processed_samples += samples_in_step
                    # Log metrics using Fabric logger
                    fabric.log_dict({
                        "loss": loss.item() * args.grad_accum_steps, # Log unnormalized loss
                        "learning_rate": current_lr,
                        "epoch": epoch + (batch_idx / actual_steps_per_epoch),
                        "step": current_grad_step,
                    }, step=current_grad_step) # Use grad step for logging

                    # Update tqdm description
                    iterable.set_postfix(
                        loss=f"{loss.item()*args.grad_accum_steps:.4f}", # Show unnormalized loss
                        lr=f"{current_lr:.2e}",
                        samples=f"{total_processed_samples}"
                    )

        # Handle any remaining gradients at the end of the epoch if num_batches % grad_accum != 0
        # Check if the last batch was an accumulation step
        if len(train_dataloader) % args.grad_accum_steps != 0:
             fabric.print("Performing final optimizer step for the epoch.")
             optimizer.step()
             optimizer.zero_grad()
             # Decide if scheduler should step here - typically yes if tied to optimizer steps
             # if total_training_steps > 0:
             #      lr_scheduler.step()


    fabric.print("Training finished.")

    # --- Saving the final model ---
    fabric.print(f"Saving final model checkpoint to {args.output_dir}")

    # Fabric's `save` handles FSDP consolidation automatically.
    # It needs a dictionary where keys are identifiers and values are the objects/state_dicts to save.
    save_path = os.path.join(args.output_dir, f"final_model_epoch_{args.epochs}.pt")

    # Create the state dictionary to save
    state = {
        "model": model,  # Pass the entire Fabric-wrapped model
        "optimizer": optimizer, # Pass the Fabric-wrapped optimizer
        "lr_scheduler": lr_scheduler.state_dict(), # Scheduler state dict is usually fine
        "args": vars(args), # Save args for reproducibility
        "epoch": args.epochs,
    }

    # Save the consolidated model checkpoint only on global rank 0
    # `fabric.save` orchestrates the gathering and saving process.
    fabric.save(save_path, state)

    # Optionally save tokenizer separately (standard HF way) on rank 0
    if fabric.global_rank == 0:
        tokenizer.save_pretrained(args.output_dir)
        fabric.print("Tokenizer saved.")
        fabric.print(f"Consolidated checkpoint saved to: {save_path}")
        fabric.print("Script finished successfully.")

if __name__ == "__main__":
    main()