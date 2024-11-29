export CUDA_VISIBLE_DEVICES=$1

method=$2 # Support PyramidKV, SnapKV, H2O, StreamingLLM, CAM, AdaKV, HeadKV
# max_capacity_prompts=$3 # 128,2048 in paper
attn_implementation=$4 # Support "flash_attention_2", "sdpa", "eager".
source_path=$5
model_path=$6
save_dir=${source_path}"results_long_bench" # path to result save_dir


for max_capacity_prompts in 64 96 128 256 512 1024 2048 4096
do
python3 run_longbench.py \
    --method ${method} \
    --model_path ${model_path} \
    --max_capacity_prompts ${max_capacity_prompts} \
    --attn_implementation ${attn_implementation} \
    --save_dir ${save_dir} \
    --use_cache True
done

