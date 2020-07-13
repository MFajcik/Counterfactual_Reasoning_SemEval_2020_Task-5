import logging
import os
import sys

import torch

from src_scripts.common.utils import setup_logging, mkdir, get_timestamp
from src_scripts.transformers_task2.ensembling.ensemble_tools import average_probs
from src_scripts.transformers_task2.task2_transformers_trainer import TransformerTask2Framework

SPLIT = "samesplit"  # samesplit, different_split
model_paths = [f"saved/{SPLIT}/{f}" for f in os.listdir(f"saved/{SPLIT}") if f.endswith(".pt")]

config = {
    "objective_variant": "independent",
    "decoding_variant": "independent",
    "tokenizer_type": "roberta-large",
    "model_type": "roberta-large",
    "max_antecedent_length": 116,  # optimized via 1000 hyperopt trials
    "max_consequent_length": 56,
    "tensorboard_logging": False,
    "model_paths": model_paths,
    "test_file": "Subtask-2-test-master/subtask2_test.csv",
    "dropout_rate": 0.,
}

_WORKDIR = ".predictions/precomputed_same_seeds_greedy_fl/test_data"

# PRECOMPUTED_PROB_FILES = None
p = ".predictions/precomputed_same_seeds_greedy_fl/test_data"
PRECOMPUTED_PROB_FILES = [f"{p}/{f}" for f in os.listdir(p) if f.endswith(".pkl")]

ensemble_prob_file = f"{_WORKDIR}/ensemble_probs_avg_{get_timestamp()}.pkl"  #
ensemble_result_file = f"{_WORKDIR}/result_{get_timestamp()}.csv"
if __name__ == "__main__":
    mkdir(_WORKDIR)
    setup_logging(os.path.basename(sys.argv[0]).split(".")[0],
                  logpath=".logs/",
                  config_path="configurations/logging.yml")
    logging.info(f"Running ensemble of {len(model_paths)} models.")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    try:
        framework = TransformerTask2Framework(config, device)
        if PRECOMPUTED_PROB_FILES is None:
            prob_files = []
            for m in config["model_paths"]:
                logging.info(f"Processing: {m}")
                framework.config["model_path"] = m
                of = f"{_WORKDIR}/" + os.path.basename(m) + "_probs.pkl"
                framework.predict_prob(file=config["test_file"],
                                       outfile=of)
                prob_files.append(of)
        else:
            prob_files = PRECOMPUTED_PROB_FILES
        average_probs(prob_files, ensemble_prob_file)
        framework.make_output(config["test_file"], ensemble_prob_file, ensemble_result_file, debug_result_file=True)
    except BaseException as be:
        logging.error(be)
        raise be
