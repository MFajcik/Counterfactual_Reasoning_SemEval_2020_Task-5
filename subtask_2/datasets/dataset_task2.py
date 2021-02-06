import csv
import json
import os
import shutil
import time
import string
import sys

import numpy as np
import pandas as pd
import torchtext.data as data
import logging

from typing import List, Tuple, Dict
from torchtext.data import RawField, Example
from tqdm import tqdm
from transformers import RobertaTokenizer, PreTrainedTokenizer, BertTokenizer, AlbertTokenizer

from configurations.task_constants import TRAIN_F_SEMEVAL2020_5, TRAIN_URL_SEMEVAL2020_5
from src_scripts.common.utils import download_url, find_sub_list
from src_scripts.tokenizers.spacy_tokenizer import tokenize, char_span_to_token_span


class SemEvalECFR_Dataset(data.Dataset):

    def __init__(self, data, fields: List[Tuple[str, data.Field]], tokenizer: PreTrainedTokenizer, model_type,
                 cachedir='.data/semeval2020_5', max_len=509, predict_mode=False, **kwargs):
        self.max_len = max_len
        self.predict_mode = predict_mode
        self.check_for_download(cachedir)
        self.tokenizer = tokenizer
        f = os.path.join(cachedir, data)
        preprocessed_f = f + f"_preprocessed_{model_type}.json"
        if not os.path.exists(preprocessed_f):
            self.debug_f = open(f".data/semeval2020_5/debug_{os.path.basename(data)}", "a+")
            s_time = time.time()
            raw_examples = self.get_example_list(f)
            self.save(preprocessed_f, raw_examples)
            logging.info(f"Dataset {preprocessed_f} created in {time.time() - s_time:.2f}s")

        s_time = time.time()
        examples = self.load(preprocessed_f, fields)
        logging.info(f"Dataset {preprocessed_f} loaded in {time.time() - s_time:.2f} s")

        super(SemEvalECFR_Dataset, self).__init__(examples, fields, **kwargs)

    def tokenize_spacy_subword(self, text):
        spacy_tokens, _ = tokenize(text)
        token_positions = []
        final_tokenized = []
        for token in spacy_tokens:
            subtokens = self.tokenizer.tokenize(token.text)
            token_accumulated_offset = 0
            final_tokenized += subtokens
            # I found no way to keep track of UNK tokens :(
            if self.tokenizer.unk_token in subtokens:
                for _ in subtokens:
                    token_positions.append((token.idx, token.idx + len(token.text)))
                continue

            for st in subtokens:
                start_ix = token.idx + token_accumulated_offset
                subtoken_len = len(st.replace('##', ''))
                end_ix = start_ix + subtoken_len
                token_positions.append([start_ix, end_ix])
                token_accumulated_offset += subtoken_len
            if not (token_positions[-1][1] == token.idx + len(token.text)):
                # Unfortunately BERT tokenizer throws away unicode symbols
                # we cannot correct keep track on words with unicode subtokens
                token_positions[-1][1] = token.idx + len(token.text)
        return final_tokenized, token_positions

    def save(self, preprocessed_f: string, raw_examples: List[Dict]):
        with open(preprocessed_f, "w") as f:
            json.dump(raw_examples, f)

    def load(self, preprocessed_f: string, fields: List[Tuple[str, RawField]]) -> List[Example]:
        with open(preprocessed_f, "r") as f:
            raw_examples = json.load(f)
            return [data.Example.fromlist([
                e["sentence_id"],
                e["raw_sentence"],
                e["raw_antecedent"],
                e["raw_consequent"],
                e["has_consequent"],
                e["sentence"],
                [1] * len(e["sentence"]),  # sentence mask
                e["sentence_positions"],
                e["antecedent_startid"],
                e["antecedent_endid"],
                e["consequent_startid"],
                e["consequent_endid"],
            ], fields) for e in raw_examples]

    @staticmethod
    def check_for_download(cachedir):
        if not os.path.exists(cachedir) and cachedir:
            os.makedirs(cachedir)
            try:
                download_url(os.path.join(cachedir, TRAIN_F_SEMEVAL2020_5), TRAIN_URL_SEMEVAL2020_5)
            except BaseException as e:
                sys.stderr.write(f'Download failed, removing directory {cachedir}\n')
                sys.stderr.flush()
                shutil.rmtree(cachedir)
                raise e

    def get_token_offsets(self, tokenized_span, tokenized_text, text_positions, ch_start, ch_end):
        answer_locations = find_sub_list(tokenized_span, tokenized_text)
        if len(answer_locations) > 1:
            # get start character offset of each span
            answer_ch_starts = [text_positions[token_span[0]][0] for token_span in
                                answer_locations]
            distance_from_gt = np.abs((np.array(answer_ch_starts) - ch_start))
            closest_match = distance_from_gt.argmin()
            token_span_start, token_span_end = answer_locations[closest_match]
        elif not answer_locations:
            # Call heuristic from AllenNLP to help :(
            token_span = char_span_to_token_span(
                text_positions,
                (ch_start, ch_end))
            token_span_start, token_span_end = token_span[0]
        else:
            token_span_start, token_span_end = answer_locations[0]
        return token_span_start, token_span_end

    def is_correct(self, gt_span, extracted_tokenized_span, e):
        def remove_ws(s):
            return "".join(s.split())

        csvf = csv.writer(self.debug_f, delimiter=',')
        if remove_ws(gt_span) != remove_ws(
                "".join(extracted_tokenized_span)):
            csvf.writerow({"sentence_id": e["sentenceID"],
                           "a_extracted": "|".join(extracted_tokenized_span),
                           "a_gt": "|".join(gt_span),
                           }.values())
            return False
        return True

    def get_example_list(self, file):
        df = pd.read_csv(file)
        examples = []
        problems = 0
        for index, row in tqdm(df.iterrows(), total=len(df), desc="Preprocessing data"):
            sentence, positions = self.tokenize_spacy_subword(row["sentence"])
            if len(sentence) > self.max_len:
                logging.info(f"Truncating sentence\n{row['sentence']}...")
            numericalized_sentence = self.tokenizer.convert_tokens_to_ids(
                sentence[:self.max_len])
            sentence_representation = self.tokenizer.build_inputs_with_special_tokens(numericalized_sentence)
            if self.predict_mode:
                ex = {
                    "sentence_id": row["sentenceID"],
                    "raw_sentence": row["sentence"],
                    "raw_antecedent": "",
                    "raw_consequent": "",
                    "sentence": sentence_representation,
                    "sentence_positions": positions,
                    "has_consequent": -1,
                    "antecedent_startid": -1,
                    "antecedent_endid": -1,
                    "consequent_startid": -1,
                    "consequent_endid": -1,
                }
            else:
                tokenized_antecedent = self.tokenize_spacy_subword(row["antecedent"])[0]

                a_s, a_e = self.get_token_offsets(tokenized_antecedent, sentence, positions, row["antecedent_startid"],
                                                  row["antecedent_endid"])
                has_consequent = row["consequent_startid"] >= 0
                if has_consequent:
                    tokenized_consequent = self.tokenize_spacy_subword(row["consequent"])[0]
                    c_s, c_e = self.get_token_offsets(tokenized_consequent, sentence, positions,
                                                      row["consequent_startid"],
                                                      row["consequent_endid"])
                else:
                    c_s, c_e = -1, -1

                if not self.is_correct(row["antecedent"], sentence[a_s:a_e + 1], row) \
                        or (has_consequent and not self.is_correct(row["consequent"], sentence[c_s:c_e + 1], row)):
                    problems += 1
                ex = {
                    "sentence_id": row["sentenceID"],
                    "raw_sentence": row["sentence"],
                    "raw_antecedent": row["antecedent"],
                    "raw_consequent": row["consequent"],
                    "sentence": sentence_representation,
                    "sentence_positions": positions,
                    "has_consequent": has_consequent,
                    "antecedent_startid": a_s,
                    "antecedent_endid": a_e,
                    "consequent_startid": c_s,
                    "consequent_endid": c_e,
                }
            examples.append(ex)
        logging.info(f"# problems: {problems}")
        logging.info(f"Problems affect {problems / len(examples) / 100:.5f} % of dataset.")
        return examples

    @staticmethod
    def prepare_fields(pad_t):
        return [
            ('sentence_id', data.RawField()),
            ('raw_sentence', data.RawField()),
            ('raw_antecedent', data.RawField()),
            ('raw_consequent', data.RawField()),
            ('has_consequent', data.Field(sequential=False, use_vocab=False, batch_first=True, is_target=True)),
            ('sentence', data.Field(batch_first=True, use_vocab=False, sequential=True, pad_token=pad_t)),
            ('sentence_padding', data.Field(use_vocab=False, batch_first=True, sequential=True, pad_token=0)),
            ('sentence_positions', data.RawField()),
            ("a_start", data.Field(sequential=False, use_vocab=False, batch_first=True, is_target=True)),
            ("a_end", data.Field(sequential=False, use_vocab=False, batch_first=True, is_target=True)),
            ("c_start", data.Field(sequential=False, use_vocab=False, batch_first=True, is_target=True)),
            ("c_end", data.Field(sequential=False, use_vocab=False, batch_first=True, is_target=True)),
        ]
