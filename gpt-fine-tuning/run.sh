export CUDA_VISIBLE_DEVICES=0
mkdir -p tests
python -m torch.distributed.launch --nproc_per_node=1 \
    --master_port 60010 clm_baseline.py \
    --dataset_name ptb_text_only \
    --dataset_config_name penn_treebank \
    --model_name_or_path gpt2 \
    --num_train_epochs 100 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --output_dir tests | tee -a tests/training.log 