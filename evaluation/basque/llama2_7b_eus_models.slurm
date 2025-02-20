#!/bin/bash
#SBATCH --job-name=eu-lmeval
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=12:00:00
#SBATCH --mem=50GB
#SBATCH --gres=gpu:1
#SBATCH --output=.slurm/llama2-7b-lmeval.out
#SBATCH --error=.slurm/llama2-7b-lmeval.err

source /gscratch5/users/asalem/environments/latest-lmeval/bin/activate 

#export HF_TOKEN=""
export HF_DATASETS_CACHE="/gscratch5/users/asalem/cache/hf_datasets_cache"
export TRANSFORMERS_CACHE="/gscratch5/users/asalem/transformers_cache/"
export TOKENIZERS_PARALLELISM=false

models=(
    "global_step1"
    "global_step2"
    "global_step3"
    "global_step4"
    "global_step5"
    "global_step6"
    "global_step7"
    "global_step8"
    "global_step9"
    "global_step10"
    "global_step30"
    "global_step50"
    "global_step80"
    "global_step100"
    "global_step200"
    "global_step300"
    "global_step400"
    "global_step500"
    "global_step600"
    "global_step700"
    "global_step900"
    "global_step1000"
    "global_step1500"
    "global_step2500"
    "global_step3500"
    "global_step4500"
    "global_step5500"
    "global_step6500"
    "global_step7500"
    "global_step8500"  
    "global_step9500"  
    "global_step10000"  
)

tasks_selected=(
    "belebele_eus_Latn"
    "eus_trivia"
    "eus_reading"
    "eus_proficiency"
    "eus_exams_eu"
    "xstorycloze_eu"
    "xnli_eu"
    "icp_bench"
)

results_path=/gscratch5/users/asalem/cpt_experiments/evaluation/basque/llama2-7b

for model in "${models[@]}"; do

    model_name='/gscratch5/users/asalem/cpt_experiments/models/'$model
    for group_name in "${tasks_selected[@]}"; do

    mkdir -p ${results_path}/${model}/${group_name}

        if [[ $group_name == "xnli_eu" || $group_name == "xstorycloze_eu" ]]; then
            num_fewshot=0
        else
            num_fewshot=5
        fi

        python3 -m lm_eval \
            --model hf \
            --model_args pretrained=$model_name,parallelize=True \
            --tasks $group_name \
            --device cuda \
            --output_path ${results_path}/${model}/${group_name}/${num_fewshot}-shot.json \
            --num_fewshot ${num_fewshot} \
            --log_samples
    done

done

