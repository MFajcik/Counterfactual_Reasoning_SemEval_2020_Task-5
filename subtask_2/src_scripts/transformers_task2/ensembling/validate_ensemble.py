import logging
import os
import sys
import numpy as np
import pandas as pd
import torch

from src_scripts.common.evaluate import evaluate_semeval2020_task5
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
    "test_file": "val_task2.csv",
    "dropout_rate": 0.,
    "cache_dir": "./.BERTcache",
}

_WORKDIR = ".predictions/precomputed_same_seeds_fl/val_data/"

timestamp = get_timestamp()
# timestamp = "2020-03-06_11:09"
ensemble_prob_file = f"{_WORKDIR}/ensemble_probs_avg_all_{timestamp}.pkl"  #
ensemble_result_file = f"{_WORKDIR}/result_{timestamp}.csv"

PRECOMPUTED_PROB_FILES = None


# p = ".predictions/precomputed_same_seeds_greedy_fl/val_data"
# PRECOMPUTED_PROB_FILES = [os.path.join(p, f) for f in os.listdir(p)]


def validate_ensemble():
    mkdir(".predictions")
    setup_logging(os.path.basename(sys.argv[0]).split(".")[0],
                  logpath=".logs/",
                  config_path="configurations/logging.yml")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    try:
        framework = TransformerTask2Framework(config, device)
        if PRECOMPUTED_PROB_FILES is None:
            prob_files = []
            for m in config["model_paths"]:
                framework.config["model_path"] = m
                of = f"{_WORKDIR}" + os.path.basename(m) + "_probs.pkl"
                framework.predict_prob(file=config["test_file"],
                                       outfile=of)
                prob_files.append(of)
        else:
            prob_files = PRECOMPUTED_PROB_FILES
        average_probs(prob_files, ensemble_prob_file)
        framework.make_output(config["test_file"], ensemble_prob_file, ensemble_result_file, debug_result_file=True)

        result = f"{ensemble_result_file}_debug.csv"
        gt = f".data/semeval2020_5/{config['test_file']}"
        gpd = pd.read_csv(gt, encoding='utf-8').to_numpy()
        rpd = pd.read_csv(result, encoding='utf-8', header=None).to_numpy()
        ids = []
        a_spans = []
        c_spans = []
        a_gt_spans = []
        c_gt_spans = []
        for g, r in zip(gpd, rpd):
            assert g[0] == r[0]
            ids.append(g[0])
            a_spans.append(r[2])
            c_spans.append("" if type(r[3]) == float and np.isnan(r[3]) else r[3])
            a_gt_spans.append(g[2])
            c_gt_spans.append("" if g[3] == '{}' else g[3])
        scores_antecedent, a_ems, a_f1s = evaluate_semeval2020_task5(dict(zip(ids, a_gt_spans)),
                                                                     dict(zip(ids, a_spans)))
        scores_consequent, c_ems, c_f1s = evaluate_semeval2020_task5(dict(zip(ids, c_gt_spans)),
                                                                     dict(zip(ids, c_spans)))

        antecedent_em, antecedent_f1 = scores_antecedent["exact_match"], scores_antecedent["f1"]
        consequent_em, consequent_f1 = scores_consequent["exact_match"], scores_consequent["f1"]
        total_em = (scores_antecedent["exact_match"] + scores_consequent["exact_match"]) / 2
        total_f1 = (scores_antecedent["f1"] + scores_consequent["f1"]) / 2

        ### Compute no-consequent accuracy
        assert len(ids) == len(c_spans) == len(c_gt_spans)
        total = 0
        hits = 0
        for _id, c_predicted, c_gt in zip(ids, c_spans, c_gt_spans):
            total += 1
            hits += int((c_gt == "" and c_predicted == "") or (c_gt != "" and c_predicted != ""))

        consequent_accuracy = 100 * hits / total

        print(f"total EM: {total_em}\n"
              f"total F1: {total_f1}\n"
              f"antecedent EM: {antecedent_em}\n"
              f"antecedent F1: {antecedent_f1}\n"
              f"consequent EM: {consequent_em}\n"
              f"consequent F1: {consequent_f1}\n"
              f"consequent ACCURACY: {consequent_accuracy}")
        print("-" * 50)
    except BaseException as be:
        logging.error(be)
        raise be


if __name__ == "__main__":
    validate_ensemble()
