import logging
import os

import torch

from atlas import dist_utils, slurm, util
from atlas.index_io import load_or_initialize_index
from atlas.model_io import create_checkpoint_directories, load_or_initialize_atlas_model
from atlas.options import get_options
from utils import dict_to_args

os.environ["TOKENIZERS_PARALLELISM"] = "true"

logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    print("WE ARE HERE!")
    options = get_options()
    opt = options.parse()
    opt.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(opt.seed)
    slurm.init_distributed_mode(opt)
    slurm.init_signal_handler()

    checkpoint_path, saved_index_path = create_checkpoint_directories(opt)

    logger = util.init_logger(
        opt.is_main, opt.is_distributed, os.path.join(checkpoint_path, "run.log")
    )
    if opt.is_main:
        options.print_options(opt)

    logger.info(f"world size: {dist_utils.get_world_size()}")

    index, passages = load_or_initialize_index(opt)
    print(f"INDEX: {index}")
    print(f"PASSAGE: {passages}")
    # model, _, _, _, _, opt, step = load_or_initialize_atlas_model(opt, eval_only=True)
    # print(model)
