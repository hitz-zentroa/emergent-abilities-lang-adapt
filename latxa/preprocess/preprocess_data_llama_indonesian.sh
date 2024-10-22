#!/bin/bash
#SBATCH -A EUHPC_E02_013
#SBATCH -p boost_usr_prod
#SBATCH --time 24:00:00         # format: HH:MM:SS
#SBATCH --nodes 1             # 4 node
#SBATCH --ntasks-per-node=4     # 4 tasks out of 32
#SBATCH --gres=gpu:4            # 4 gpus per node out of 4
#SBATCH --mem=494000MB              # memory per node out of 494000MB
#SBATCH --output=.slurm/preprocess_indonesian.out
#SBATCH --error=.slurm/preprocess_indonesian.err
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

# Move to the gpt-neox install
cd /leonardo_work/EUHPC_E02_013/gpt-neox-main/gpt-neox

# setup the virtual env
source ${WORK}/environments/neox-env/bin/activate

# Loop over each subfolder
for dir in /leonardo_scratch/large/userexternal/amohamed/indonesian_data/* ; do
    if [ -d "$dir" ]; then
        # Get the name of the subfolder
        subfolder=${dir##*/}

        # if bne_v1, skip
        if [ "$subfolder" == "bne_v1" ]; then
            continue
        fi

        echo "Processing ${subfolder}"

        # Create the output directory
        mkdir -p "/leonardo_scratch/large/userexternal/amohamed/indonesian_data_preprocessed/${subfolder}"

        # Define the data types
        splits=("train")

        # Loop over the data types
        for split in "${splits[@]}"; do
            echo "Processing ${split}"

            # Calculate the number of lines in the file
            num_lines=$(cat "/leonardo_scratch/large/userexternal/amohamed/indonesian_data/${subfolder}/${split}.jsonl" | wc -l)
	    
	        echo ${num_lines}
            # Preprocess the data
            python tools/datasets/preprocess_data.py \
                --input "/leonardo_scratch/large/userexternal/amohamed/indonesian_data/${subfolder}/${split}.jsonl" \
                --output-prefix "/leonardo_scratch/large/userexternal/amohamed/indonesian_data_preprocessed/${subfolder}/${split}" \
                --tokenizer-type "SPMTokenizer" \
                --vocab-file "/leonardo/home/userexternal/amohamed/tokenizer.model" \
                --num-docs $num_lines \
                --append-eod \
                --workers 8
        done
    fi
done
