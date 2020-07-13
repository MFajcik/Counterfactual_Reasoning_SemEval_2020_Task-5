import logging
import os
import sys
import torch

from src_scripts.common.utils import setup_logging
from src_scripts.transformers_task2.task2_transformers_trainer import TransformerTask2Framework

config = {
    "objective_variant": "independent",
    "decoding_variant": "independent",
    "tokenizer_type": "albert-xxlarge-v2",
    "model_type": "albert-xxlarge-v2",
    "max_iterations": 200,
    "batch_size": 1,
    "true_batch_size": 32,
    "max_grad_norm": 4.,
    "lookahead_optimizer": True,
    "lookahead_K": 10,
    "lookahead_alpha": 0.5,
    "max_answer_length": 50,
    "optimizer": "adam",
    "learning_rate": 3.5e-6 * 1.5,
    "validation_batch_size": 4,
    "tensorboard_logging": True,
    "eval_only": False
}

if __name__ == "__main__":
    setup_logging(os.path.basename(sys.argv[0]).split(".")[0],
                  logpath=".logs/",
                  config_path="configurations/logging.yml")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    try:
        framework = TransformerTask2Framework(config, device)
        framework.fit()
    except BaseException as be:
        logging.error(be)
        raise be
