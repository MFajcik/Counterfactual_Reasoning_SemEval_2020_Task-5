import pandas as pd
import numpy as np

from src_scripts.common.evaluate import evaluate_semeval2020_task5

if __name__ == "__main__":
    result = ".predictions/result_2020-03-03_12:55.csv_debug.csv"
    gt = ".data/semeval2020_5/val_task2.csv"
    gpd = pd.read_csv(gt, encoding='utf-8').to_numpy()
    rpd = pd.read_csv(result, encoding='utf-8', header=None).to_numpy()
    ids = []
    a_spans = []
    c_spans = []
    a_gt_spans = []
    c_gt_spans = []
    for g, r in zip(gpd, rpd):
        ids.append(g[0])
        assert g[0] == r[0]
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

    print(f"total EM: {total_em}\n"
          f"total F1: {total_f1}\n"
          f"antecedent EM: {antecedent_em}\n"
          f"antecedent F1: {antecedent_f1}\n"
          f"consequent EM: {consequent_em}\n"
          f"consequent F1: {consequent_f1}")
    print("-" * 50)
