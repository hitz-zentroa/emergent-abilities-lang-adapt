cd gpt-neox/
pip install -r requirements/requirements.txt
pip install -r requirements/requirements-wandb.txt
pip install -r requirements/requirements-flashattention.txt
python ./megatron/fused_kernels/setup.py install # optional, if using fused kernels


