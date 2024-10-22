inpath=/leonardo_scratch/large/userexternal/amohamed
#inpath=/leonardo_scratch/large/userexternal/jetxaniz
path=/leonardo_scratch/large/userexternal/amohamed
#/leonardo_scratch/large/userexternal/jetxaniz/Llama-2-7b-neox-eus-v1.1
input=indonesian_ckpts/Llama-2-7b-neox-indonesian-english/global_step
output=indonesian_ckpts/Llama-2-7b-hf-indonesian-english/global_step

source ${WORK}/environments/neox-env-2/bin/activate

# merge yml configs in ${path}/${input}${i}/configs into ${path}/${input}${i}/config.yml
python ${HOME}/latxa/convert/merge_configs.py \
    --input_dir ${inpath}/${input}1000/configs \
    --output_dir ${path}/config.yml

cd ${HOME}/latxa_rerun_itziar/leonardo/gpt-neox/tools/

# loop global step with step 100
for i in {500..7000..500}; do
    echo "Converting ${inpath}/${input}${i} to ${path}/${output}${i}"
    python convert_llama_sequential_to_hf.py \
        --input_dir ${inpath}/${input}${i} \
        --config_file ${path}/config.yml \
        --output_dir ${path}/${output}${i} 
done
