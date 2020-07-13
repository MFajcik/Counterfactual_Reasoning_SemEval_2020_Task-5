import csv
import pickle

from transformers import RobertaTokenizer

from configurations.task_constants import TRAIN_F_SEMEVAL2020_5
from datasets.dataset_task2 import SemEvalECFR_Dataset

if __name__ == "__main__":
    _RANDOM_STATE_PATH = ".randomstate/random_state.pkl"
    tokenizer = RobertaTokenizer.from_pretrained("roberta-large", cache_dir="./.BERTcache")
    fields = SemEvalECFR_Dataset.prepare_fields(pad_t=tokenizer.pad_token_id)

    data = SemEvalECFR_Dataset(TRAIN_F_SEMEVAL2020_5, fields=fields, tokenizer=tokenizer,
                               model_type="roberta-large")

    with open(_RANDOM_STATE_PATH, "rb") as f:
        random_state = pickle.load(f)
    _, val = data.split(split_ratio=0.9, random_state=random_state)
    with open(".data/semeval2020_5/val_task2.csv", 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["sentenceID", "sentence", "antecedent", "consequent", "antecedent_startid", "antecedent_endid",
                         "consequent_startid", "consequent_endid"])
        for e in val:
            writer.writerow([
                e.sentence_id,
                e.raw_sentence,
                e.raw_antecedent,
                e.raw_consequent,
                e.a_start,
                e.a_end,
                e.c_start,
                e.c_end
            ])
    # sentenceID,sentence,antecedent,consequent,antecedent_startid,antecedent_endid,consequent_startid,consequent_endid
    # 200007,"If that was my daughter, I would have asked If I did something wrong.",If that was my daughter,I would have asked If I did something wrong,0,22,25,67
