import logging
import os
import pickle
import sys
from random import random, shuffle

import pandas as pd
import torch
import numpy as np
from torchtext.data import Iterator
from tqdm import tqdm

from datasets.dataset_task2 import SemEvalECFR_Dataset
from src_scripts.common.decoding import decode_wth_hacks, get_spans
from src_scripts.common.evaluate import evaluate_semeval2020_task5
from src_scripts.common.utils import mkdir, setup_logging
from src_scripts.transformers_task2.ensembling.ensemble_tools import average_probs
from src_scripts.transformers_task2.task2_transformers_trainer import TransformerTask2Framework

_WDIR = ".ensemble_experiments"
config = {
    "objective_variant": "independent",
    "decoding_variant": "independent",
    "tokenizer_type": "roberta-large",
    "model_type": "roberta-large",
    "max_antecedent_length": 116,  # optimized via 1000 hyperopt trials
    "max_consequent_length": 56,
    "tensorboard_logging": False,
    "model_paths": [],
    "test_file": "val_task2.csv",
    "dropout_rate": 0.,
}


def find_best_ensemble_greedy(precomputed_files_dir, task_framework):
    files = sorted(
        [os.path.join(precomputed_files_dir, f)
         for f in os.listdir(precomputed_files_dir)
         if f.startswith("checkpoint") and
         f.endswith(".pkl")])
    # preload probs
    probs = []
    for in_f in tqdm(files, desc="Loading probabilities..."):
        with open(in_f, "rb") as f:
            probs.append(pickle.load(f))
    prob_dict = dict(zip(files, probs))

    all_models = set(files)
    ensemble_models = set()
    best_score = 0
    available_models = all_models - ensemble_models

    while len(available_models) > 0:
        available_models = all_models - ensemble_models
        print(best_score)
        found = False
        topickfrom = (list(available_models))
        shuffle(topickfrom)
        for ix, m in tqdm(enumerate(topickfrom)):
            # if ix > 2:
            #    break
            candidate_models = ensemble_models.copy()
            candidate_models.add(m)
            prob_files = list(candidate_models)
            probs = [prob_dict[pf] for pf in prob_files]
            new_score = evaluate_ensemble(prob_files=prob_files,
                                          framework=task_framework,
                                          probs=probs)

            if new_score > best_score:
                best_score = new_score
                ensemble_models.add(m)
                found = True
                break
        if not found:
            break

    # logging.info('\n'.join(ensemble_matrices))
    # logging.info(f"F1: {best_F1}")
    return best_score, ensemble_models


#
# def remove_worst_k(k=10):
#     files = sorted(os.listdir("saved/ensemble/numpy"))
#     valid = [f for f in files if f.startswith("val_") and f.endswith("npy")]
#     result_files = sorted([f for f in valid if "result" in f], reverse=True)
#     # print(result_files)
#     label_file = [f for f in valid if "labels" in f][0]
#     labels = np.load(os.path.join("saved/ensemble/numpy", label_file))
#     result_matrices = {result_file: np.load(os.path.join("saved/ensemble/numpy", result_file)) for result_file in
#                        result_files}
#
#     ensemble_matrices = set(result_matrices.keys())
#     best_F1 = find_F1(labels, [result_matrices[m] for m in ensemble_matrices])
#     removed = 0
#     to_remove = None
#     while removed < k:
#         for m in ensemble_matrices:
#             F1 = find_F1(labels, [result_matrices[m] for m in ensemble_matrices - {m}])
#             if F1 > best_F1:
#                 best_F1 = F1
#                 to_remove = m
#         if to_remove is not None:
#             ensemble_matrices.remove(to_remove)
#             print(f"Removed :{to_remove}")
#             # print(f"Best F1{best_F1}")
#             to_remove = None
#             removed += 1
#         else:
#             break
#
#     # logging.info('\n'.join(ensemble_matrices))
#     # logging.info(f"F1: {best_F1}")
#     return best_F1, ensemble_matrices

gt = f".data/semeval2020_5/{config['test_file']}"
gpd = pd.read_csv(gt, encoding='utf-8').to_numpy()

predict_iter = None


def get_res(framework, averaged):
    out_list = []
    ensemble_probs = averaged
    global predict_iter
    if predict_iter is None:
        fields = SemEvalECFR_Dataset.prepare_fields(pad_t=framework.tokenizer.pad_token_id)
        data = SemEvalECFR_Dataset(config['test_file'], fields=fields, tokenizer=framework.tokenizer,
                                   model_type=framework.config["model_type"], predict_mode=True)
        predict_iter = Iterator(data,
                                sort=False,
                                batch_size=1,
                                train=False,
                                repeat=False)

    for i, batch in enumerate(predict_iter):
        assert batch.batch_size == 1
        # 0 th element is sentinel representing no-answer
        id = batch.sentence_id[0]
        logprobs_antecedent_S, logprobs_antecedent_E, logprobs_consequent_S, logprobs_consequent_E = \
            torch.FloatTensor(ensemble_probs[id]["antecedent_S"]).log().unsqueeze(0), \
            torch.FloatTensor(ensemble_probs[id]["antecedent_E"]).log().unsqueeze(0), \
            torch.FloatTensor(ensemble_probs[id]["consequent_S"]).log().unsqueeze(0), \
            torch.FloatTensor(ensemble_probs[id]["consequent_E"]).log().unsqueeze(0)

        decode_f = lambda s, e, max_len, mask_sentinel: decode_wth_hacks(s, e, has_sentinel=mask_sentinel,
                                                                         hacks={
                                                                             "max_answer_length": max_len,
                                                                             "combine_surface_forms": (
                                                                                 False, None)
                                                                         })
        candidates_a = \
            decode_f(logprobs_antecedent_S, logprobs_antecedent_E, framework.config["max_antecedent_length"],
                     False)[1]
        candidates_c = \
            decode_f(logprobs_consequent_S, logprobs_consequent_E, framework.config["max_consequent_length"],
                     True)[1]
        a_span = get_spans(candidates_a, batch.sentence_positions, batch.raw_sentence)[0]
        c_span = get_spans(candidates_c, batch.sentence_positions, batch.raw_sentence,
                           no_answer_sentinel=True)[0]

        candidates_c = (candidates_c[0] - 1, candidates_c[1] - 1)  # c includes sentinel at start

        get_character_offset_from_token_offset = lambda token_offset: batch.sentence_positions[0][
            token_offset] if token_offset > -1 else [-1, -1]

        # sentenceID 	antecedent_startid 	antecedent_endid 	consequent_startid 	consequent_endid
        # apply -1 to end_ids because, the character span is inclusive
        out_list.append([
            id,
            batch.raw_sentence[0],
            a_span,
            c_span,
            get_character_offset_from_token_offset(candidates_a[0].item())[0],
            get_character_offset_from_token_offset(candidates_a[1].item())[1] - 1,
            get_character_offset_from_token_offset(candidates_c[0].item())[0],
            get_character_offset_from_token_offset(candidates_c[1].item())[1] - 1 if
            get_character_offset_from_token_offset(candidates_c[1].item())[1] > -1 else -1,
        ])
    return np.array(out_list)


gpd = pd.read_csv(gt, encoding='utf-8').to_numpy()


def evaluate_ensemble(prob_files, framework, probs,
                      ensemble_prob_file=f"{_WDIR}/greedy_ens_p.pkl",
                      ensemble_result_file=f"{_WDIR}/greedy_ens_r.csv",
                      validation_file=""):
    averaged = average_probs(prob_files, ensemble_prob_file, preloaded_probs=probs, return_not_save=True)
    rpd = get_res(framework, averaged)
    total_em = evaluate(gpd, rpd)
    return total_em


def evaluate(gpd, rpd):
    ids = []
    a_spans = []
    c_spans = []
    a_gt_spans = []
    c_gt_spans = []
    # skip header

    for g, r in zip(gpd, rpd):
        assert g[0] == int(r[0])  # ids must match
        ids.append(g[0])
        a_spans.append(r[2])
        c_spans.append("" if type(r[3]) == float and np.isnan(r[3]) else r[3])
        a_gt_spans.append(g[2])
        c_gt_spans.append("" if g[3] == '{}' else g[3])
    scores_antecedent, a_ems, a_f1s = evaluate_semeval2020_task5(dict(zip(ids, a_gt_spans)),
                                                                 dict(zip(ids, a_spans)))
    scores_consequent, c_ems, c_f1s = evaluate_semeval2020_task5(dict(zip(ids, c_gt_spans)),
                                                                 dict(zip(ids, c_spans)))
    total_em = (scores_antecedent["exact_match"] + scores_consequent["exact_match"]) / 2
    return total_em


if __name__ == "__main__":
    mkdir(_WDIR)
    setup_logging(os.path.basename(sys.argv[0]).split(".")[0],
                  logpath=".logs/",
                  config_path="configurations/logging.yml")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    framework = TransformerTask2Framework(config, device)

    logger = logging.getLogger()
    NL = '\n'
    best_from_best = 0
    for _ in range(50 * 6):
        logger.disabled = True
        best_score, ensemble_models = find_best_ensemble_greedy(
            ".predictions/precomputed_same_seeds_fl/val_data", framework)
        logger.disabled = False

        if best_score > best_from_best:
            best_from_best = best_score

        logging.info(f"BEST_SCORE: {best_score}")
        logging.info(f"MODELS:\n{NL.join(ensemble_models)}")
        logging.info(f"#" * 50)
    logging.info(f"BEST SCORE IS: {best_from_best}")
