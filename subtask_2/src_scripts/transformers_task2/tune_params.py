import copy
import json
import logging
import math
import os
import pickle
import sys
import torch

from src_scripts.common.utils import setup_logging
from hyperopt import hp, fmin, STATUS_OK, Trials, tpe, STATUS_FAIL

from src_scripts.transformers_task2.task2_transformers_trainer import TransformerTask2Framework

config = {
    "objective_variant": "independent",
    "decoding_variant": "independent",

    "tokenizer_type": "roberta-large",
    "model_type": "roberta-large",

    "max_iterations": 200,

    "batch_size": 1,
    "true_batch_size": 32,
    "max_grad_norm": 4.,

    "learning_rate": 3.5e-6,
    "lookahead_optimizer": True,
    "lookahead_K": 10,
    "lookahead_alpha": 0.5,
    "max_antecedent_length": 50,
    "max_consequent_length": 50,
    "optimizer": "adam",
    "validation_batch_size": 4,
    "weight_decay": 0.01,

    "patience": 4,
    "tensorboard_logging": False,

    "model_path": "",
    "eval_only": True
}

optimized_config = {
    "dropout_rate": 0.04152933951236242,
    "learning_rate": 1.26299972676114e-05,
    "lookahead_K": 7.0,
    "lookahead_alpha": 0.46995660710569487,
    "max_grad_norm": 7.738899649787129,
    "true_batch_size": 64,
    "weight_decay": 0.02088116729288835
}
config.update(optimized_config)


class FitWrapper:
    def __init__(self, framework, default_config, fw_inst=None, fw_args=None):
        self.fw = framework
        self.default_config = default_config
        self.fw_inst = fw_inst
        self.fw_args = fw_args


def obj(fit_wrapper, opt_config):
    try:
        config = copy.deepcopy(fit_wrapper.default_config)
        config.update(opt_config)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        logger = logging.getLogger()
        logger.disabled = True
        framework = fit_wrapper.fw(config, device)
        stats = framework.fit()
        logger.disabled = False

        assert "status" not in stats and "loss" not in stats
        stats['status'] = STATUS_OK
        stats['loss'] = -stats["em"]
        logging.info(json.dumps(stats, indent=4, sort_keys=True))

    except BaseException as be:
        logging.error(be)
        print(be)
        return {'loss': math.inf, 'status': STATUS_FAIL}

    return stats


def obj2(fit_wrapper, opt_config):
    try:
        config = copy.deepcopy(fit_wrapper.default_config)
        config.update(opt_config)
        logger = logging.getLogger()
        logger.disabled = True
        fit_wrapper.fw_inst.config = config
        stats = fit_wrapper.fw_inst._fit(*fit_wrapper.fw_args)
        logger.disabled = False

        assert "status" not in stats and "loss" not in stats
        stats['status'] = STATUS_OK
        stats['loss'] = -stats["em"]
        logging.info(json.dumps(stats, indent=4, sort_keys=True))

    except BaseException as be:
        logging.error(be)
        print(be)
        return {'loss': math.inf, 'status': STATUS_FAIL}

    return stats


if __name__ == "__main__":
    from hyperopt import hp
    from hyperopt.pyll.base import scope

    setup_logging(os.path.basename(sys.argv[0]).split(".")[0],
                  logpath="..logs/",
                  config_path="configurations/logging.yml")
    # space = {"dropout_rate": hp.uniform("dropout_rate", low=0.0, high=0.6),
    #          "true_batch_size": hp.choice("true_batch_size", [16, 32, 40, 64]),
    #          "learning_rate": hp.uniform("learning_rate", low=1.5e-6, high=5e-5),
    #          "lookahead_K": scope.int(hp.quniform('lookahead_K', 3, 15, 1)),
    #          "lookahead_alpha": hp.uniform("lookahead_alpha", low=0.2, high=0.8),
    #          "max_grad_norm": hp.uniform("max_grad_norm", low=2., high=10.),
    #          "weight_decay": hp.uniform("weight_decay", low=0.000, high=1e-1)
    #          }
    space = {
        "max_antecedent_length": scope.int(hp.quniform('max_antecedent_length', 30, 300, 1)),
        "max_consequent_length": scope.int(hp.quniform('max_consequent_length', 30, 300, 1)),
    }

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    framework = TransformerTask2Framework(config, device)
    args = framework.preload()

    wrapper = FitWrapper(framework=TransformerTask2Framework, default_config=config, fw_inst=framework, fw_args=args)

    trials = Trials()
    best = fmin(lambda a: obj2(wrapper, a),
                space=space,
                algo=tpe.suggest,
                max_evals=1000,
                trials=trials)
    logging.info(best)
    with open("roberta-large-hyperopt_best.pkl", "wb") as f:
        pickle.dump(best, f)
    with open("roberta-large-hyperopt_trials.pkl", "wb") as f:
        pickle.dump(trials, f)
