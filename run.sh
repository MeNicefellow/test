python train.py \
    --accelerator=cuda \
    --devices=2 \
    --strategy=fsdp \
    --precision=bf16-mixed \
    --batch_size_per_device=2 \
    --grad_accum_steps=8 \
    --model_name_or_path=gpt2 \
    --output_dir=./fabric_fsdp_output