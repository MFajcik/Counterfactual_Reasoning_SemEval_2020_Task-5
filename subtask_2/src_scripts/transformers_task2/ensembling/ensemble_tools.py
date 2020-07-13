import logging
import pickle
import numpy as np

from tqdm import tqdm


def average_probs(infiles, outfile, preloaded_probs=None, return_not_save=False):
    if preloaded_probs:
        probs = preloaded_probs
    else:
        probs = []
        for in_f in tqdm(infiles, desc="Loading probabilities..."):
            with open(in_f, "rb") as f:
                probs.append(pickle.load(f))
    ids = probs[0].keys()
    logging.info(f"averaging {len(probs)} probability distributions")
    logging.info(f"number of items: {len(ids)}")
    antecedent_k = ["antecedent_S", "antecedent_E"]
    consequent_k = ["consequent_S", "consequent_E"]
    averaged = dict()
    for id in ids:
        # models x 2 x len
        antecedent_probd = np.stack([np.stack([p[id][key] for key in antecedent_k]) for p in probs])
        consequent_probd = np.stack([np.stack([p[id][key] for key in consequent_k]) for p in probs])
        # 2 x len
        antecedent_probd = antecedent_probd.mean(axis=0)
        consequent_probd = consequent_probd.mean(axis=0)
        averaged[id] = {
            "antecedent_S": antecedent_probd[0],
            "antecedent_E": antecedent_probd[1],
            "consequent_S": consequent_probd[0],
            "consequent_E": consequent_probd[1],
        }
    if return_not_save:
        return averaged
    with open(outfile, "wb") as of_handle:
        pickle.dump(averaged, of_handle)

