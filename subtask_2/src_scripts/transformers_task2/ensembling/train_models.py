import logging
import os
import sys
import torch

from src_scripts.common.utils import setup_logging
from src_scripts.transformers_task2.task2_transformers_trainer import TransformerTask2Framework

config = {
    "objective_variant": "independent",
    "decoding_variant": "independent",
    "tokenizer_type": "roberta-large",
    "model_type": "roberta-large",
    "max_iterations": 200,
    "batch_size": 1,
    "true_batch_size": 32,
    "max_grad_norm": 4.,
    "lookahead_optimizer": True,
    "lookahead_K": 10,
    "lookahead_alpha": 0.5,
    "optimizer": "adam",
    "learning_rate": 3.5e-6,
    "validation_batch_size": 4,
    "tensorboard_logging": False,
    "patience": 4,
    "eval_only": False
}

optimized_config = {
    "dropout_rate": 0.04152933951236242,
    "learning_rate": 1.26299972676114e-05,
    "lookahead_K": 7.0,
    "lookahead_alpha": 0.46995660710569487,
    "max_grad_norm": 7.738899649787129,
    "true_batch_size": 64,
    "weight_decay": 0.02088116729288835,
    "max_antecedent_length": 116,  # optimized via 1000 hyperopt trials
    "max_consequent_length": 56,
}

config.update(optimized_config)
MODELS = 10
if __name__ == "__main__":
    for i in range(5, 8):
        for j in range(10):
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            for _ in range(MODELS):
                setup_logging(os.path.basename(sys.argv[0]).split(".")[0],
                              logpath=".logs/",
                              config_path="configurations/logging.yml")
                try:
                    framework = TransformerTask2Framework(config, device,
                                                          random_state_path=f".randomstate/random_state_{i}.pkl")
                    framework.fit()
                except BaseException as be:
                    logging.error(be)
                    raise be
