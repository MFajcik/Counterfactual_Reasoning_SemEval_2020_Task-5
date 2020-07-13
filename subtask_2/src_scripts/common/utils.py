import datetime
import logging.config
import os
import socket
import sys
from urllib import request

import yaml


def download_url(path, url):
    sys.stderr.write(f'Downloading from {url} into {path}\n')
    sys.stderr.flush()
    request.urlretrieve(url, path)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def report_parameters(model):
    num_pars = {name: p.numel() for name, p in model.named_parameters() if p.requires_grad}
    num_sizes = {name: p.shape for name, p in model.named_parameters() if p.requires_grad}
    return num_pars, num_sizes


def get_timestamp():
    return datetime.datetime.now().strftime('%Y-%m-%d_%H:%M')


def mkdir(s):
    if not os.path.exists(s):
        os.makedirs(s)


def touch(f):
    """
    Create empty file at given location f
    :param f: path to file
    """
    basedir = os.path.dirname(f)
    if not os.path.exists(basedir):
        os.makedirs(basedir)
    open(f, 'a').close()

def find_sub_list(sl, l):
    results = []
    sll = len(sl)
    if sll < 1:
        return results
    for ind in (i for i, e in enumerate(l) if e == sl[0]):
        if l[ind:ind + sll] == sl:
            results.append((ind, ind + sll - 1))

    return results


class LevelOnly(object):
    levels = {
        "CRITICAL": 50,
        "ERROR": 40,
        "WARNING": 30,
        "INFO": 20,
        "DEBUG": 10,
        "NOTSET": 0,
    }

    def __init__(self, level):
        self.__level = self.levels[level]

    def filter(self, logRecord):
        return logRecord.levelno <= self.__level


def setup_logging(
        module,
        default_level=logging.INFO,
        env_key='LOG_CFG',
        logpath=os.getcwd(),
        extra_name="",
        config_path=None
):
    """
        Setup logging configuration\n
        Logging configuration should be available in `YAML` file described by `env_key` environment variable

        :param module:     name of the module
        :param logpath:    path to logging folder [default: script's working directory]
        :param config_path: configuration file, has more priority than configuration file obtained via `env_key`
        :param env_key:    evironment variable containing path to configuration file
        :param default_level: default logging level, (in case of no local configuration is found)
    """

    if not os.path.exists(os.path.dirname(logpath)):
        os.makedirs(os.path.dirname(logpath))
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    stamp = timestamp + "_" + socket.gethostname() + "_" + extra_name

    path = config_path if config_path is not None else os.getenv(env_key, None)
    if path is not None and os.path.exists(path):
        with open(path, 'rt') as f:
            config = yaml.safe_load(f.read())
            for h in config['handlers'].values():
                if h['class'] == 'logging.FileHandler':
                    h['filename'] = os.path.join(logpath, module, stamp, h['filename'])
                    touch(h['filename'])
            for f in config['filters'].values():
                if '()' in f:
                    f['()'] = globals()[f['()']]
            logging.config.dictConfig(config)
    else:
        logging.basicConfig(level=default_level, filename=os.path.join(logpath, stamp))
