import logging
from collections import Counter
import string
import re


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def evaluate_semeval2020_task5(gts, predictions):
    f1 = exact_match = total = 0
    ems = []
    f1s = []
    for e_id, ground_truth in gts.items():
        total += 1
        if e_id not in predictions:
            message = 'Unanswered question ' + e_id + \
                      ' will receive score 0.'
            logging.error(message)
            continue
        prediction = predictions[e_id]
        if ground_truth == "":
            if prediction == "":
                exact_match += 1
                f1 += 1
            ems.append(int(prediction == "")), f1s.append(float(prediction == ""))
        else:
            local_em = int(metric_max_over_ground_truths(
                exact_match_score, prediction, [ground_truth]))
            local_f1 = metric_max_over_ground_truths(
                f1_score, prediction, [ground_truth])
            ems.append(local_em), f1s.append(local_f1)
            exact_match += local_em
            f1 += local_f1
    exact_match = 100.0 * exact_match / total
    f1 = 100.0 * f1 / total

    return {'exact_match': exact_match, 'f1': f1}, ems, f1s
