# -*- coding: utf-8 -*-

import sys
import logging


def get_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if logger.hasHandlers():
        return logger
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    handler.setFormatter(fmt=logging.Formatter(
        fmt='%(asctime)s [%(name)s] [%(levelname)s] >>> %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    ))
    logger.addHandler(handler)
    return logger
