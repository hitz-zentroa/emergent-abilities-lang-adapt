from argparse import ArgumentParser
import glob
import os
import re
from tqdm import tqdm
import torch

IGNORED_MODEL_STATE_KEYS = [
    "buffer_names",
    "optimizer",
    "param_shapes",
    "frozen_param_shapes",
    "shared_params",
    "frozen_param_fragments",
    "lr_scheduler",
    "data_sampler",
    "random_ltd",
    "sparse_tensor_module_names",
    "global_steps",
    "ds_config",
    "ds_version",
    "args",
    "random_rng_state",
    "np_rng_state",
    "torch_rng_state",
    "cuda_rng_state",
    "rng_tracker_states",
]

# LAYER_NAME_TEMPLATE = "layer_{layer_id}-model_{rank_id}-model_states.pt"
LAYER_NAME_RE = re.compile(r"layer_(?P<layer>\d+)-model_(?P<rank>\d+)-model_states.pt")
MP_RANK_NAME_TEMPLATE = "mp_rank_{rank_id}_model_states.pt"


def load_model_state(rank, input_dir, num_shards):
    file_name = os.path.join(
        input_dir, MP_RANK_NAME_TEMPLATE.format(rank_id=f"{rank:02d}")
    )
    model_state = {
        key: value
        for key, value in torch.load(file_name, map_location="cpu").items()
        if key not in IGNORED_MODEL_STATE_KEYS
    }
    model_state["mp_world_size"] = num_shards
    model_state["dp_world_size"] = 1
    model_state["module"] = {}

    return model_state


def main(args):
    model_states = [
        load_model_state(rank, input_dir=args.input_dir, num_shards=args.num_shards)
        for rank in range(args.num_shards)
    ]

    layer_files = glob.glob(os.path.join(args.input_dir, "layer_*"))
    for file_name in (pbar := tqdm(layer_files, desc="Loading...")):
        pbar.set_description(f"Loading: {os.path.basename(file_name)}")
        layer, rank = tuple(map(int, LAYER_NAME_RE.match(os.path.basename(file_name)).groups()))
        for key, value in torch.load(file_name, map_location="cpu").items():
            model_states[rank]["module"][f"sequential.{layer}.{key}"] = value

    os.makedirs(args.output_dir, exist_ok=True)
    for rank in tqdm(
        range(args.num_shards), total=args.num_shards, desc="Saving to files"
    ):
        torch.save(
            model_states[rank],
            os.path.join(
                args.output_dir, MP_RANK_NAME_TEMPLATE.format(rank_id=f"{rank:02d}")
            ),
        )


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument(
        "--input_dir",
        type=str,
        default="scratch_julen/Llama-2-70b-neox-TP-4-PP/global_step0",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="scratch/Llama-2-70b-neox-TP-4",
    )
    parser.add_argument("--num_shards", type=int, default=4)

    args = parser.parse_args()
    main(args)
