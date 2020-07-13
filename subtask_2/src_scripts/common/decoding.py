import math

import numpy as np
import torch
import torch.nn.functional as F

from src_scripts.common.utils import find_sub_list
from scipy.special import softmax


def get_spans(candidates, token_positions, raw_text, no_answer_sentinel=False):
    r = []
    if no_answer_sentinel:
        candidates = (candidates[0] - 1, candidates[1] - 1)
    for i in range(len(raw_text)):
        candidate_start = candidates[0][i]
        candidate_end = candidates[1][i]

        if no_answer_sentinel and \
                -1 == candidate_start == candidate_end:
            r.append("")
            continue

        if candidate_start > len(token_positions[i]) - 1:
            candidate_start = len(token_positions[i]) - 1
        if candidate_end > len(token_positions[i]) - 1:
            candidate_end = len(token_positions[i]) - 1

        r.append(raw_text[i][token_positions[i][candidate_start][0]:
                             token_positions[i][candidate_end][-1]])
    return r


def decode(span_start_logits: torch.Tensor, span_end_logits: torch.Tensor, score="probs") -> \
        (torch.Tensor, torch.Tensor):
    """
    This method has been borrowed from AllenNLP
    :param span_start_logits:
    :param span_end_logits:
    :return:
    """
    # We call the inputs "logits" - they could either be unnormalized logits or normalized log
    # probabilities.  A log_softmax operation is a constant shifting of the entire logit
    # vector, so taking an argmax over either one gives the same result.
    if span_start_logits.dim() != 2 or span_end_logits.dim() != 2:
        raise ValueError("Input shapes must be (batch_size, passage_length)")
    batch_size, passage_length = span_start_logits.size()
    device = span_start_logits.device
    # (batch_size, passage_length, passage_length)
    span_log_probs = span_start_logits.unsqueeze(2) + span_end_logits.unsqueeze(1)
    # Only the upper triangle of the span matrix is valid; the lower triangle has entries where
    # the span ends before it starts.
    span_log_mask = torch.triu(torch.ones((passage_length, passage_length),
                                          device=device)).log().unsqueeze(0)
    valid_span_log_probs = span_log_probs + span_log_mask

    # Here we take the span matrix and flatten it, then find the best span using argmax.  We
    # can recover the start and end indices from this flattened list using simple modular
    # arithmetic.
    # (batch_size, passage_length * passage_length)
    valid_span_log_probs = valid_span_log_probs.view(batch_size, -1)
    if score == "probs":
        valid_span_scores = F.softmax(valid_span_log_probs, dim=-1)
    elif score == "logprobs":
        valid_span_scores = valid_span_log_probs
    else:
        raise NotImplemented(f"Unknown score type \"{score}\"")

    best_span_scores, best_spans = valid_span_scores.max(-1)
    span_start_indices = best_spans // passage_length
    span_end_indices = best_spans % passage_length

    return best_span_scores, (span_start_indices, span_end_indices)


def decode_wth_hacks(span_start_logits: torch.Tensor,
                     span_end_logits: torch.Tensor,
                     has_sentinel=False,
                     score="probs", hacks={
            "max_answer_length": 30,
            "combine_surface_forms": (False, None)
        }) -> \
        (torch.Tensor, torch.Tensor):
    """
    This method has been borrowed from AllenNLP
    :param span_start_logits:
    :param span_end_logits:
    :return:
    """
    # We call the inputs "logits" - they could either be unnormalized logits or normalized log
    # probabilities.  A log_softmax operation is a constant shifting of the entire logit
    # vector, so taking an argmax over either one gives the same result.
    # has_sentinel = False
    if "combine_surface_forms" not in hacks:
        hacks["combine_surface_forms"] = (False, None)
    if hacks["combine_surface_forms"][0]:
        assert score == "probs"
        assert hacks["combine_surface_forms"][1] is not None
    if span_start_logits.dim() != 2 or span_end_logits.dim() != 2:
        raise ValueError("Input shapes must be (batch_size, passage_length)")
    batch_size, passage_length = span_start_logits.size()
    device = span_start_logits.device
    # (batch_size, passage_length, passage_length)
    span_log_probs = span_start_logits.unsqueeze(2) + span_end_logits.unsqueeze(1)
    if has_sentinel:
        span_log_probs[:, 1:, 0] = -math.inf
        span_log_probs[:, 0, 1:] = -math.inf
    # Only the upper triangle of the span matrix is valid; the lower triangle has entries where
    # the span ends before it starts.
    span_log_mask = torch.triu(torch.ones((passage_length, passage_length),
                                          device=device)).log().unsqueeze(0)
    valid_span_log_probs = span_log_probs + span_log_mask

    spans_longer_than_maxlen_mask = torch.Tensor([[j - i + 1 > hacks["max_answer_length"]
                                                   for j in range(passage_length)] for i in
                                                  range(passage_length)]) \
        .to(valid_span_log_probs.get_device()
            if valid_span_log_probs.get_device() > -1 else torch.device(
        "cpu"))
    valid_span_log_probs.masked_fill_(spans_longer_than_maxlen_mask.unsqueeze(0).bool(), -math.inf)

    # Here we take the span matrix and flatten it, then find the best span using argmax.  We
    # can recover the start and end indices from this flattened list using simple modular
    # arithmetic.
    # (batch_size, passage_length * passage_length)
    valid_span_log_probs = valid_span_log_probs.view(batch_size, -1)
    if score == "probs":
        valid_span_scores = F.softmax(valid_span_log_probs, dim=-1)
    elif score == "logprobs":
        valid_span_scores = valid_span_log_probs
    else:
        raise NotImplemented(f"Unknown score type \"{score}\"")
    if hacks["combine_surface_forms"][0]:
        # Re-ranking top-N based on surface form
        sorted_probs, indices = torch.sort(valid_span_scores, dim=-1, descending=True)
        span_start_indices = indices // passage_length
        span_end_indices = indices % passage_length

        N = 100  # top-N surface form reranking
        sorted_probs, span_start_indices, span_end_indices = sorted_probs[:, :N], \
                                                             span_start_indices[:, :N], \
                                                             span_end_indices[:, :N]
        if type(hacks["combine_surface_forms"][1]) == torch.Tensor:
            hacks["combine_surface_forms"] = hacks["combine_surface_forms"][0], hacks["combine_surface_forms"][
                1].tolist()

        for i in range(len(span_start_indices)):
            processed = []
            for a, e in zip(span_start_indices[i].tolist(), span_end_indices[i].tolist()):
                if (a, e) in processed:
                    continue
                processed.append((a, e))  # do not adjust value of other spans with this span
                span_occurences = find_sub_list(hacks["combine_surface_forms"][1][i][a:e + 1],
                                                hacks["combine_surface_forms"][1][i])
                if len(span_occurences) > 1:
                    for span in span_occurences:
                        if span in processed:
                            continue
                        processed.append(span)  # do not adjust value of zeroed result
                        valid_span_scores[i, a * passage_length + e] += valid_span_scores[i,
                                                                                          span[0] * passage_length +
                                                                                          span[1]]
                        valid_span_scores[i,
                                          span[0] * passage_length + span[1]] = 0.

    best_span_scores, best_spans = valid_span_scores.max(-1)
    span_start_indices = best_spans // passage_length
    span_end_indices = best_spans % passage_length

    return best_span_scores, (span_start_indices, span_end_indices)