# This script is adapted from 
# https://github.com/FranxYao/Long-Context-Data-Engineering.git


mkdir -p ./results_needle/logs/
mkdir -p ./results_needle/img/


METHOD='pyramidkv'       # ['full', 'pyramidkv', 'snapkv', 'streamingllm', 'h2o']
MAX_CAPACITY_PROMPT=96  # [64, 96, 128, 256, 512, 1024, 2048, ...]
attn_implementation="flash_attention_2" # Support "flash_attention_2", "sdpa", "None".
TAG=test


# For Llama3-8b

(
python -u run_needle_in_haystack.py --s_len 1000 --e_len 8001\
    --model_provider LLaMA3 \
    --model_name YOU_PATH_TO_LLAMA_3 \
    --attn_implementation ${attn_implementation} \
    --step 100 \
    --method $METHOD \
    --max_capacity_prompt $MAX_CAPACITY_PROMPT \
    --model_version LlaMA3_${METHOD}_${MAX_CAPACITY_PROMPT}_${TAG}
) 2>&1  | tee results_needle/logs/LlaMA3_${METHOD}_${MAX_CAPACITY_PROMPT}_${TAG}.log



# For Mistral

# (
# python -u run_needle_in_haystack.py --s_len 400 --e_len 32001\
#     --model_provider Mistral \
#     --model_name YOU_PATH_TO_MISTRAL_2 \
#     --step 400 \
#     --method $METHOD \
#     --max_capacity_prompt $MAX_CAPACITY_PROMPT \
#     --model_version Mistral2_${METHOD}_${MAX_CAPACITY_PROMPT}_${TAG}
# ) 2>&1  | tee logs/Mistral2_${METHOD}_${MAX_CAPACITY_PROMPT}_${TAG}.log