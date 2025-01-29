inpath=/leonardo_scratch/large/userexternal/amohamed
#inpath=/leonardo_scratch/large/userexternal/jetxaniz
path=/leonardo_scratch/large/userexternal/amohamed
#/leonardo_scratch/large/userexternal/jetxaniz/Llama-2-7b-neox-eus-v1.1
input=Llama-2-7b-neox-eus-ema_17112024/global_step
output=Llama-2-7b-hf-eus-ema_17112024/global_step

source ${WORK}/environments/neox-env/bin/activate

# merge yml configs in ${path}/${input}${i}/configs into ${path}/${input}${i}/config.yml
python ${HOME}/latxa_rerun_itziar/leonardo/latxa/convert/merge_configs.py \
    --input_dir ${inpath}/${input}5500/configs \
    --output_dir ${path}/config.yml

cd ${HOME}/latxa_rerun_itziar/leonardo/gpt-neox/tools/

# loop global step with step 100
for i in {5500..5500..500}; do
    echo "Converting ${inpath}/${input}${i} to ${path}/${output}${i}"
    python convert_llama_sequential_to_hf.py \
        --input_dir ${inpath}/${input}${i} \
        --config_file ${path}/config.yml \
        --output_dir ${path}/${output}${i} 
done
