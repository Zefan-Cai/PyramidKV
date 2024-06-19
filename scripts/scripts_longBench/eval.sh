export CUDA_VISIBLE_DEVICES=0


model_path=""
method="PyramidKV" # Support PyramidKV, SnapKV, StreamingLLM, H2O
max_capacity_prompts=512 # 128,2048 in paper
attn_implementation="flash_attention_2" # Support "flash_attention_2", "sdpa", "".
save_dir="results_long_bench" # path to result save_dir

python3 run_longbench.py \
    --method ${method} \
    --model_path ${model_path} \
    --max_capacity_prompts ${max_capacity_prompts} \
    --attn_implementation ${attn_implementation} \
    --save_dir ${save_dir} \
    --use_cache True
