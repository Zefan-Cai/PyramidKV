export CUDA_VISIBLE_DEVICES=0

# base_dir: path to PyramidKV
base_dir=""

# model ckpt need to put at {base_dir}/ckpt/{model_name}.

model_name=Llama-3-8B-Instruct
method="PyramidKV"
max_capacity_prompts=512

# path to result save_dir
save_dir="results"

# results will be put at {base_dir}/{results}.

python3 ${base_dir}/run.py \
    --base_dir ${base_dir} \
    --method ${method} \
    --model_name ${model_name} \
    --max_capacity_prompts ${max_capacity_prompts} \
    --save_dir ${save_dir} \
    --use_cache True
