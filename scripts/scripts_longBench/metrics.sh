
base_dir=$1
mode_name=$2
results_dir=$3
# /home/caizf/projects/Attention
# /mnt/users/v-caizefan/Attention


python3 ${base_dir}/PyramidQA/eval_ssp.py \
    --model_name ${mode_name} \
    --results_dir ${results_dir}
