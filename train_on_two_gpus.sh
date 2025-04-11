deepspeed --num_gpus 2 train_gpt2_deepspeed.py \
    --deepspeed \
    --deepspeed_config ds_config_fsdp.json \
    --model_name_or_path gpt2 \
    --output_dir ./ds_gpt2_output_run1 \
    --epochs 1
