path=/leonardo_scratch/large/userexternal/amohamed/
cd ${WORK}/gpt-neox/

source ${WORK}/environments/neox-env/bin/activate

cd ${HOME}/latxa_rerun_itziar/leonardo/gpt-neox/tools

for TP in 1 2 4; do
    python convert_raw_llama_weights_to_neox.py \
        --input_dir /leonardo_scratch/large/userexternal/jetxaniz/Llama-2-7b/ \
        --model_size 7B \
        --output_dir ${path}Llama-2-7b-neox-TP-${TP} \
        --num_output_shards ${TP}
done
