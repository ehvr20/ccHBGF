import logging

from ._spec import ccHBGF

__all__ = ['ccHBGF']

# Setup package logger
logger = logging.getLogger('ccHBGF')
logger.setLevel(logging.WARNING)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(name)s - %(asctime)s: %(message)s', datefmt='%H:%M:%S')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)
logger.propagate = False
