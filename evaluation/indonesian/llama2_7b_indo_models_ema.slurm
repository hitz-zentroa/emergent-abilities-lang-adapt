#!/bin/bash
#SBATCH --job-name=indo-lmeval
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=24:00:00
#SBATCH --mem=50GB
#SBATCH --gres=gpu:1
#SBATCH --output=.slurm/llama2_7b_indo.out
#SBATCH --error=.slurm/llama2_7b_indo.err


source /gscratch5/users/asalem/lmeval-env2/bin/activate

export HF_DATASETS_CACHE="/gscratch5/users/asalem/hf_datasets_cache"
export TRANSFORMERS_CACHE="/gscratch5/users/asalem/transformers_cache/"
export TOKENIZERS_PARALLELISM=false

models=(
    "ema/global_step10000"
    "ema-curr/global_step10000"
)

tasks_selected=(
    "indonesian_mmlu"
    "icp_bench"
)

results_path=/gscratch5/users/asalem/cpt_experiments/evaluation/indo/llama2-7b

for model in "${models[@]}"; do

    model_name='/gscratch5/users/asalem/cpt_experiments/indo_models/'$model

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

