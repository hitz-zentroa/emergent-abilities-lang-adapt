#!/bin/bash
#SBATCH --job-name=ar-lmeval
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=24:00:00
#SBATCH --mem=50GB
#SBATCH --gres=gpu:1
#SBATCH --output=.slurm/llama2_7b_arabic.out
#SBATCH --error=.slurm/llama2_7b_arabic.err


source /gscratch5/users/asalem/lmeval-env2/bin/activate

export HF_DATASETS_CACHE="/gscratch5/users/asalem/hf_datasets_cache"
export TRANSFORMERS_CACHE="/gscratch5/users/asalem/transformers_cache/"
export TOKENIZERS_PARALLELISM=false

models=(
    "ar-curr/global_step10000"
)

tasks_selected=(
    "arabicmmlu"
    "icp_bench"
)

results_path=/gscratch5/users/asalem/cpt_experiments/evaluation/arabic/llama2-7b

for model in "${models[@]}"; do

    model_name='/gscratch5/users/asalem/cpt_experiments/arabic_models/'$model

    for group_name in "${tasks_selected[@]}"; do

    mkdir -p ${results_path}/${model_name}/${group_name}

        python3 -m lm_eval \
            --model hf \
            --model_args pretrained=$model_name,parallelize=True \
            --tasks $group_name \
            --device cuda \
            --output_path ${results_path}/${model}/${group_name}/results.json \
            --log_samples

    done

done

