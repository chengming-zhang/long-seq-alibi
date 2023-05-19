export CUDA_VISIBLE_DEVICES=0,1,2,3
max_len=2048
block_size=2048
mkdir -p tests
python -m torch.distributed.launch --nproc_per_node=4 \
    --master_port 60010 clm_pos_long_seq_fine_tune.py  --pos_embed "alibi" \
    --max_length ${max_len} --block_size ${block_size} --overwrite_cache True \
    --dataset_name ptb_text_only \
    --dataset_config_name penn_treebank \
    --model_name_or_path gpt2 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --output_dir tests | tee -a tests/training.log 