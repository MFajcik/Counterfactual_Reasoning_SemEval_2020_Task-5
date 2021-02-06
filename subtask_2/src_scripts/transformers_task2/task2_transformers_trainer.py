import csv
import json
import logging
import math
import os
import pickle
import random
import socket
import time
from typing import Tuple, Union

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torchtext.data import BucketIterator, Iterator
from tqdm import tqdm
from transformers import RobertaTokenizer, AlbertTokenizer, BertTokenizer

from configurations.task_constants import TRAIN_F_SEMEVAL2020_5
from datasets.dataset_task2 import SemEvalECFR_Dataset
from models.task2_transformer import TransformerForECFR
from optimizers.lookahead import Lookahead
from src_scripts.common.decoding import get_spans, decode_wth_hacks
from src_scripts.common.evaluate import evaluate_semeval2020_task5
from src_scripts.common.utils import mkdir, count_parameters, report_parameters, get_timestamp


class TransformerTask2Framework:
    def __init__(self, config, device, random_state_path=".randomstate/random_state.pkl"):
        mkdir("results")
        mkdir("saved")
        mkdir(".randomstate")
        self._RANDOM_STATE_PATH = random_state_path
        if not os.path.isfile(self._RANDOM_STATE_PATH):
            state = random.getstate()
            with open(self._RANDOM_STATE_PATH, "wb") as f:
                pickle.dump(state, f)

        self.config = config
        self.device = device
        self.n_iter = 0
        if config["tokenizer_type"].startswith("bert"):
            self.tokenizer = BertTokenizer.from_pretrained(config["tokenizer_type"], cache_dir=config["cache_dir"])

        if config["tokenizer_type"].startswith("roberta"):
            self.tokenizer = RobertaTokenizer.from_pretrained(config["tokenizer_type"], cache_dir=config["cache_dir"])

        if config["tokenizer_type"].startswith("albert"):
            self.tokenizer = AlbertTokenizer.from_pretrained(config["tokenizer_type"], cache_dir=config["cache_dir"])

        if config["tensorboard_logging"]:
            from torch.utils.tensorboard import SummaryWriter
            self.boardwriter = SummaryWriter()

    def train_epoch(self, model: TransformerForECFR, optimizer: torch.optim.Optimizer,
                    scheduler: Union[torch.optim.lr_scheduler.LambdaLR, None], train_iter: Iterator) -> float:
        model.train()
        train_loss = 0
        optimizer.zero_grad()
        update_ratio = self.config["true_batch_size"] // self.config["batch_size"]
        updated = False
        it = tqdm(train_iter, total=len(train_iter.data()) // train_iter.batch_size + 1)
        for i, batch in enumerate(it):
            updated = False
            # 0 th element is sentinel representing no-answer
            batch.c_start += 1
            batch.c_end += 1

            if self.config["objective_variant"] == "independent":
                # consequent null score = logits_consequent_S[0] + logits_consequent_S[0]
                logits_antecedent_S, logits_antecedent_E, logits_consequent_S, logits_consequent_E = model(
                    batch)
                loss_as, loss_ae, \
                loss_cs, loss_ce = self.masked_cross_entropy(logits_antecedent_S, batch.a_start), \
                                   self.masked_cross_entropy(logits_antecedent_E, batch.a_end), \
                                   self.masked_cross_entropy(logits_consequent_S, batch.c_start), \
                                   self.masked_cross_entropy(logits_consequent_E, batch.c_end)
                loss = (loss_as + loss_ae + loss_cs + loss_ce) / 4.
            elif self.config["objective_variant"] == "compound":
                antecedent_joint_logits, consequent_joint_logits, \
                logits_antecedent_S, logits_antecedent_E, \
                logits_consequent_S, logits_consequent_E = model(batch)
                loss_as, loss_ae, \
                loss_cs, loss_ce = self.masked_cross_entropy(logits_antecedent_S, batch.a_start), \
                                   self.masked_cross_entropy(logits_antecedent_E, batch.a_end), \
                                   self.masked_cross_entropy(logits_consequent_S, batch.c_start), \
                                   self.masked_cross_entropy(logits_consequent_E, batch.c_end)

                loss_joint_a = self.masked_cross_entropy(
                    antecedent_joint_logits.view(antecedent_joint_logits.shape[0], -1),
                    batch.a_start * logits_antecedent_E.shape[-1] + batch.a_end)

                loss_joint_c = self.masked_cross_entropy(
                    consequent_joint_logits.view(consequent_joint_logits.shape[0], -1),
                    batch.c_start * logits_consequent_E.shape[-1] + batch.c_end)

                loss = (loss_as + loss_ae + loss_cs + loss_ce + loss_joint_a + loss_joint_c) / 6.

            loss.backward()
            if (i + 1) % update_ratio == 0:
                self.n_iter += 1
                torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()),
                                               self.config["max_grad_norm"])

                if scheduler is not None:
                    scheduler.step()
                optimizer.step()
                optimizer.zero_grad()
                updated = True
                tr_loss = train_loss / (i + 1)
                it.set_description(f"Training loss: {tr_loss}")

                if self.config["tensorboard_logging"] and self.n_iter % 100 == 0:  # Log tboard after every 100 updates
                    self.boardwriter.add_scalar('Online_Loss/train', tr_loss, global_step=self.n_iter)
                    self.boardwriter.add_scalar('Stochastic_Loss/train', loss.item(), global_step=self.n_iter)
                    # grad norms w.r.expectation_embeddings_unilm input dimensions
                    for name, param in model.named_parameters():
                        if param.grad is not None and not param.grad.data.is_sparse:
                            self.boardwriter.add_histogram(f"gradients_wrt_hidden_{name}/",
                                                           param.grad.data.norm(p=2, dim=0),
                                                           global_step=self.n_iter)

            train_loss += loss.item()

        # Do the last step if needed with what has been accumulated
        if not updated:
            if scheduler is not None:
                scheduler.step()
            optimizer.step()
            optimizer.zero_grad()
        return train_loss / len(train_iter.data())

    @staticmethod
    def masked_cross_entropy(input, unclamped_target, reduce=True, mask_infs=True):
        """ Deals with ignored indices and inf losses - which are result of early masking"""
        # Indices longer than input are clamped to ignore index and ignored
        ignored_index = input.size(1)
        target = unclamped_target.clamp(0, ignored_index)

        losses = F.cross_entropy(input, target, ignore_index=ignored_index, reduction="none")
        # get rid of inf losses
        if mask_infs:
            losses = losses.masked_fill(losses == math.inf, 0.)
        if reduce:
            return torch.sum(losses) / (torch.nonzero(losses.data).size(0) + 1e-15)
        else:
            return losses

    @torch.no_grad()
    def validate(self, model: TransformerForECFR, iter: Iterator, log_results=False) -> \
            Tuple[float, float, float, float, float, float, float, float]:
        model.eval()

        ids = []
        lossvalues = []
        a_spans = []
        c_spans = []
        a_gt_spans = []
        c_gt_spans = []
        a_span_probs = []
        c_span_probs = []
        sentences = []
        for i, batch in tqdm(enumerate(iter), total=len(iter.data()) // iter.batch_size + 1):
            # 0 th element is sentinel representing no-answer
            batch.c_start += 1
            batch.c_end += 1

            ids += batch.sentence_id
            sentences += batch.raw_sentence

            if self.config["objective_variant"] == "independent":
                logits_antecedent_S, logits_antecedent_E, logits_consequent_S, logits_consequent_E = model(batch)
                loss_as, loss_ae, \
                loss_cs, loss_ce = self.masked_cross_entropy(logits_antecedent_S, batch.a_start, reduce=False), \
                                   self.masked_cross_entropy(logits_antecedent_E, batch.a_end, reduce=False), \
                                   self.masked_cross_entropy(logits_consequent_S, batch.c_start, reduce=False), \
                                   self.masked_cross_entropy(logits_consequent_E, batch.c_end, reduce=False)
                loss = (loss_as + loss_ae + loss_cs + loss_ce) / 4.

            elif self.config["objective_variant"] == "compound":
                antecedent_joint_logits, consequent_joint_logits, \
                logits_antecedent_S, logits_antecedent_E, \
                logits_consequent_S, logits_consequent_E = model(batch)

                loss_as, loss_ae, \
                loss_cs, loss_ce = self.masked_cross_entropy(logits_antecedent_S, batch.a_start, reduce=False), \
                                   self.masked_cross_entropy(logits_antecedent_E, batch.a_end, reduce=False), \
                                   self.masked_cross_entropy(logits_consequent_S, batch.c_start, reduce=False), \
                                   self.masked_cross_entropy(logits_consequent_E, batch.c_end, reduce=False)

                loss_joint_a = self.masked_cross_entropy(
                    antecedent_joint_logits.view(antecedent_joint_logits.shape[0], -1),
                    batch.a_start * logits_antecedent_E.shape[-1] + batch.a_end)

                loss_joint_c = self.masked_cross_entropy(
                    consequent_joint_logits.view(consequent_joint_logits.shape[0], -1),
                    batch.c_start * logits_consequent_E.shape[-1] + batch.c_end)

                loss = (loss_as + loss_ae + loss_cs + loss_ce + loss_joint_a + loss_joint_c) / 6.

            lossvalues += loss.tolist()

            if self.config["decoding_variant"] == "independent":
                best_span_probs_a, candidates_a = decode_wth_hacks(logits_antecedent_S, logits_antecedent_E, hacks={
                    "max_answer_length": self.config["max_antecedent_length"],
                    "combine_surface_forms": (False, None)
                })
                best_span_probs_c, candidates_c = decode_wth_hacks(logits_consequent_S, logits_consequent_E,
                                                                   has_sentinel=True, hacks={
                        "max_answer_length": self.config["max_consequent_length"],
                        "combine_surface_forms": (False, None)
                    })

            a_span_probs += best_span_probs_a.tolist()
            a_spans += get_spans(candidates_a, batch.sentence_positions, batch.raw_sentence)

            a_gt_spans += batch.raw_antecedent

            c_span_probs += best_span_probs_c.tolist()
            c_spans += get_spans(candidates_c, batch.sentence_positions, batch.raw_sentence, no_answer_sentinel=True)
            c_gt_spans += [conseq_text if has_conseq else ""
                           for conseq_text, has_conseq in zip(batch.raw_consequent, batch.has_consequent)]

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

        if log_results:
            self.log_results(fn=f"EM_{total_em}_F1_{total_f1}_results", ids=ids, a_ems=a_ems, a_f1s=a_f1s, c_ems=c_ems,
                             c_f1s=c_f1s,
                             a_spans=a_spans, a_gt_spans=a_gt_spans, c_spans=c_spans, c_gt_spans=c_gt_spans,
                             sentences=sentences)

        return sum(lossvalues) / len(
            lossvalues), total_em, total_f1, antecedent_em, antecedent_f1, consequent_em, consequent_f1, consequent_accuracy

    @torch.no_grad()
    def predict_prob(self, file, outfile):
        logging.debug(json.dumps(self.config, indent=4, sort_keys=True))
        fields = SemEvalECFR_Dataset.prepare_fields(pad_t=self.tokenizer.pad_token_id)

        data = SemEvalECFR_Dataset(file, fields=fields, tokenizer=self.tokenizer, cachedir="",
                                   model_type=self.config["model_type"], predict_mode=True)
        predict_iter = Iterator(data,
                                sort=False,
                                batch_size=1,
                                train=False,
                                repeat=False,
                                device=self.device)
        state_dict = torch.load(self.config["model_path"])
        model: TransformerForECFR = TransformerForECFR(self.config,
                                                       sep_token=self.tokenizer.sep_token_id,
                                                       pad_token=self.tokenizer.pad_token_id)
        model.load_state_dict(state_dict)
        model = model.to(self.device)

        start_time = time.time()
        try:
            model.eval()
            distributions = dict()
            for i, batch in tqdm(enumerate(predict_iter),
                                 total=len(predict_iter.data()) // predict_iter.batch_size + 1):
                assert batch.batch_size == 1
                # debug_input = " ".join(self.tokenizer.convert_ids_to_tokens(batch.sentence[0]) )
                # 0 th element is sentinel representing no-answer
                distributions[batch.sentence_id[0]] = dict()
                logits_antecedent_S, logits_antecedent_E, logits_consequent_S, logits_consequent_E = model(batch)
                softmax_np = lambda x: F.softmax(x[0], dim=-1).cpu().numpy()
                probs_antecedent_S, probs_antecedent_E, \
                probs_consequent_S, probs_consequent_E = \
                    softmax_np(logits_antecedent_S), softmax_np(logits_antecedent_E), \
                    softmax_np(logits_consequent_S), softmax_np(logits_consequent_E)
                distributions[batch.sentence_id[0]]["antecedent_S"] = probs_antecedent_S
                distributions[batch.sentence_id[0]]["antecedent_E"] = probs_antecedent_E
                distributions[batch.sentence_id[0]]["consequent_S"] = probs_consequent_S
                distributions[batch.sentence_id[0]]["consequent_E"] = probs_consequent_E
            with open(outfile, "wb") as of_handle:
                pickle.dump(distributions, of_handle)
        finally:
            logging.info(f'Finished after {(time.time() - start_time) / 60} minutes.')

    # noinspection PyUnresolvedReferences
    def fit(self):
        model, optimizer, train_iter, val_iter = self.preload()
        return self._fit(model, optimizer, train_iter, val_iter)

    def _fit(self, model, optimizer, train_iter, val_iter):
        start_time = time.time()
        try:
            best_val_loss = math.inf
            no_improvement_steps = 0
            best_val_f1 = 0
            best_em = 0
            for it in range(self.config["max_iterations"]):
                logging.info(f"Iteration {it}")

                if not self.config["eval_only"]:
                    self.train_epoch(model, optimizer, scheduler=None, train_iter=train_iter)

                validation_loss, em, f1, antecedent_em, antecedent_f1, consequent_em, consequent_f1, consequent_acc = \
                    self.validate(model, val_iter, log_results=self.config["eval_only"])
                logging.info(f"validation loss: {validation_loss}\n")
                logging.info(f"total EM: {em}\n"
                             f"total F1: {f1}\n"
                             f"antecedent EM: {antecedent_em}\n"
                             f"antecedent F1: {antecedent_f1}\n"
                             f"consequent EM: {consequent_em}\n"
                             f"consequent F1: {consequent_f1}\n"
                             f"no-consequent ACCURACY: {consequent_acc}")
                logging.info("-" * 50)

                if self.config["eval_only"]:
                    return {"em": em, "f1": f1, "val_loss": validation_loss}
                # if validation_loss < best_val_loss: best_val_loss = validation_loss
                # if f1 > best_val_f1: best_val_f1 = f1
                if em > best_em:
                    no_improvement_steps = 0
                    best_em = em
                    best_val_f1 = f1
                    best_val_loss = validation_loss
                    if em > 70:
                        # Do all this on CPU, this is memory exhaustive!
                        model.to(torch.device("cpu"))
                        torch.save(model.state_dict(),
                                   f"saved/checkpoint"
                                   f"_{self.config['model_type']}"
                                   f"_{str(self.__class__)}"
                                   f"_EM_{em:.2f}_F1_{f1:.2f}_L_{validation_loss:.2f}_{get_timestamp()}"
                                   f"_{socket.gethostname()}_state_dict.pt")
                        model.to(self.device)
                else:
                    no_improvement_steps += 1
                logging.info(f"BEST L/F1/EM = {best_val_loss:.2f}/{best_val_f1:.2f}/{best_em:.2f}")
                if no_improvement_steps > self.config["patience"]:
                    logging.info("Running out of patience!")
                    break

        except KeyboardInterrupt:
            logging.info('-' * 120)
            logging.info('Exit from training early.')
        finally:
            logging.info(f'Finished after {(time.time() - start_time) / 60} minutes.')
        return {"em": best_em, "f1": best_val_f1, "val_loss": best_val_loss}

    def preload(self):
        logging.debug(json.dumps(self.config, indent=4, sort_keys=True))
        fields = SemEvalECFR_Dataset.prepare_fields(pad_t=self.tokenizer.pad_token_id)
        data = SemEvalECFR_Dataset(TRAIN_F_SEMEVAL2020_5, fields=fields, tokenizer=self.tokenizer,
                                   model_type=self.config["model_type"])
        with open(self._RANDOM_STATE_PATH, "rb") as f:
            random_state = pickle.load(f)
        train, val = data.split(split_ratio=0.9, random_state=random_state)
        train_iter = BucketIterator(train, sort_key=lambda x: -(len(x.sentence)),
                                    shuffle=True, sort=False,
                                    batch_size=self.config["batch_size"], train=True,
                                    repeat=False,
                                    device=self.device)
        val_iter = BucketIterator(val, sort_key=lambda x: -(len(x.sentence)),
                                  shuffle=False, sort=True,
                                  batch_size=self.config["validation_batch_size"], train=False,
                                  repeat=False,
                                  device=self.device)

        def load_model() -> TransformerForECFR:
            return TransformerForECFR(self.config,
                                      sep_token=self.tokenizer.sep_token_id,
                                      pad_token=self.tokenizer.pad_token_id)

        if "model_path" in self.config:
            m = torch.load(self.config["model_path"])
            state_dict = m.state_dict() if hasattr(m, "state_dict") else m
            model = load_model()
            model.load_state_dict(state_dict)
        else:
            model = load_model()
        model = model.to(self.device)
        model.config = self.config
        logging.info(f"Models has {count_parameters(model)} parameters")
        param_sizes, param_shapes = report_parameters(model)
        param_sizes = "\n'".join(str(param_sizes).split(", '"))
        param_shapes = "\n'".join(str(param_shapes).split(", '"))
        logging.debug(f"Model structure:\n{param_sizes}\n{param_shapes}\n")
        if self.config["optimizer"] == "adam" or self.config["optimizer"] == "adamw":
            optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                              lr=self.config["learning_rate"], weight_decay=self.config["weight_decay"])
        else:
            raise NotImplementedError(f"Option {self.config['optimizer']} for \"optimizer\" setting is undefined.")
        if self.config["lookahead_optimizer"]:
            optimizer = Lookahead(optimizer, k=self.config["lookahead_K"], alpha=self.config["lookahead_alpha"])
        return model, optimizer, train_iter, val_iter

    def log_results(self, fn, ids, a_ems, a_f1s, c_ems, c_f1s, a_spans, a_gt_spans, c_spans, c_gt_spans, sentences):
        with open(f'results/{fn}.csv', 'w') as csvfile:
            logwriter = csv.writer(csvfile)
            header = ["correct", "ids", "a_em", "a_f1", "c_em", "c_f1", "a_gt", "a_pred", "c_gt", "c_pred", "sentence"]
            logwriter.writerow(header)
            itemlists = [ids, a_ems, a_f1s, c_ems, c_f1s, a_gt_spans, a_spans, c_gt_spans, c_spans, sentences]
            itemlistlens = [len(i) for i in itemlists]
            for i in itemlistlens: assert i == itemlistlens[0]
            corrects = [int(items[1] == items[2] == items[3] == items[4] == 1) for items in zip(*itemlists)]
            logwriter.writerows(zip(*([corrects] + itemlists)))

    def make_output(self, file, ensemble_prob_file, ensemble_result_file, debug_result_file=True, averaged=None):
        """
        Submission format for task2
        https://competitions.codalab.org/competitions/21691#learn_the_details-evaluation

        If there is no consequent part (a consequent part not always exists in a counterfactual statement) in this sentence, please put '-1' in the consequent_startid and 'consequent_endid'. The 'sentenceID' should be in the same order as in 'test.csv' for subtask-2 (in evaluation phase).
        sentenceID 	antecedent_startid 	antecedent_endid 	consequent_startid 	consequent_endid
        104975 	    15 	                72 	                88 	                100
        104976 	    18 	                38 	                -1              	-1
        ... 	... 	... 	... 	...

        :param ensemble_prob_file:
        :param ensemble_result_file:
        :return:
        """
        header = ["sentenceID", "antecedent_startid", "antecedent_endid", "consequent_startid", "consequent_endid"]
        with open(ensemble_result_file, 'w') as csvfile:
            result_writer = csv.writer(csvfile)
            result_writer.writerow(header)
            if averaged is not None:
                ensemble_probs = averaged
            else:
                with open(ensemble_prob_file, "rb") as f:
                    ensemble_probs = pickle.load(f)
            fields = SemEvalECFR_Dataset.prepare_fields(pad_t=self.tokenizer.pad_token_id)

            data = SemEvalECFR_Dataset(file, fields=fields, tokenizer=self.tokenizer, cachedir='',
                                       model_type=self.config["model_type"], predict_mode=True)
            predict_iter = Iterator(data,
                                    sort=False,
                                    batch_size=1,
                                    train=False,
                                    repeat=False,
                                    device=self.device)
            start_time = time.time()
            dbg_f = None
            try:
                if debug_result_file:
                    dbg_f = open(ensemble_result_file + "_readable.csv", 'w')
                    debug_result_writer = csv.writer(dbg_f)
                distributions = dict()
                for i, batch in tqdm(enumerate(predict_iter),
                                     total=len(predict_iter.data()) // predict_iter.batch_size + 1):
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
                        decode_f(logprobs_antecedent_S, logprobs_antecedent_E, self.config["max_antecedent_length"],
                                 False)[1]
                    candidates_c = \
                        decode_f(logprobs_consequent_S, logprobs_consequent_E, self.config["max_consequent_length"],
                                 True)[1]
                    a_span = get_spans(candidates_a, batch.sentence_positions, batch.raw_sentence)[0]
                    c_span = get_spans(candidates_c, batch.sentence_positions, batch.raw_sentence,
                                       no_answer_sentinel=True)[0]

                    candidates_c = (candidates_c[0] - 1, candidates_c[1] - 1)  # c includes sentinel at start

                    get_character_offset_from_token_offset = lambda token_offset: batch.sentence_positions[0][
                        token_offset] if token_offset > -1 else [-1, -1]

                    # sentenceID 	antecedent_startid 	antecedent_endid 	consequent_startid 	consequent_endid
                    # apply -1 to end_ids because, the character span is inclusive
                    result_writer.writerow([
                        id,
                        get_character_offset_from_token_offset(candidates_a[0].item())[0],
                        get_character_offset_from_token_offset(candidates_a[1].item())[1] - 1,
                        get_character_offset_from_token_offset(candidates_c[0].item())[0],
                        get_character_offset_from_token_offset(candidates_c[1].item())[1] - 1 if
                        get_character_offset_from_token_offset(candidates_c[1].item())[1] > -1 else -1,
                    ])
                    debug_result_writer.writerow([
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
            finally:
                if dbg_f is not None:
                    dbg_f.close()
                logging.info(f'Finished after {(time.time() - start_time) / 60} minutes.')
