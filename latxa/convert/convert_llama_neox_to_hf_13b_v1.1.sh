#!/bin/bash
#SBATCH -A EUHPC_E02_013
#SBATCH -p boost_usr_prod
#SBATCH --time 1:00:00         # format: HH:MM:SS
#SBATCH --nodes 1               # 4 node
#SBATCH --ntasks-per-node=4     # 4 tasks out of 32
#SBATCH --gres=gpu:4            # 4 gpus per node out of 4
#SBATCH --mem=494000MB              # memory per node out of 494000MB
#SBATCH --output=.slurm/latxa-13b-v1.1_ema.out
#SBATCH --error=.slurm/latxa-13b-v1.1_ema.err
#SBATCH --exclusive
#SBATCH --requeue

#--open-mode=append
# load leonardo modules
module load profile/deeplrn
module load python/3.10.8--gcc--11.3.0
module load cuda/11.8
module load openmpi/4.1.4--gcc--11.3.0-cuda-11.8
module load zlib/1.2.13--gcc--11.3.0
module load git-lfs

source ${WORK}/environments/neox-env-2/bin/activate


path=/leonardo_scratch/large/userexternal/amohamed
input=Llama-2-13b-neox-eus-v1.1-ema/global_step
output=Llama-2-13b-hf-eus-v1.1-ema/global_step


# merge yml configs in ${path}/${input}${i}/configs into ${path}/${input}${i}/config.yml
python ${HOME}/latxa/convert/merge_configs.py \
    --input_dir ${path}/${input}8000/configs \
    --output_dir ${path}/config.yml

cd ${HOME}/latxa_rerun_itziar/leonardo/gpt-neox/tools/

# loop global step with step 100
for i in {8000..8000..100}; do
    echo "Converting ${path}/${input}${i} to ${path}/${output}${i}"
    python convert_llama_sequential_to_hf.py \
        --input_dir ${path}/${input}${i} \
        --config_file ${path}/config.yml \
        --output_dir ${path}/${output}${i} 
done
