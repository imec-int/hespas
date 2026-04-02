# Copyright (c) 2026 imec
# SPDX-License-Identifier: MIT
import shutil
import logging
from functools import lru_cache
from bisect import bisect

logging.PROGRESS = logging.INFO + 1
logging.addLevelName(logging.PROGRESS, "PROGRESS")
logging.progress = lambda msg, *args, **kwargs: logging.log(logging.PROGRESS, msg, *args, **kwargs)
logging.RESULTS = logging.PROGRESS + 1
logging.addLevelName(logging.RESULTS, "RESULTS")
logging.results = lambda msg, *args, **kwargs: logging.log(logging.RESULTS, msg, *args, **kwargs)

class LogFormatter(logging.Formatter):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        non_verbose_formats = {
            logging.INFO: '[%(levelname)s] %(message)s',
            logging.RESULTS: '%(message)s',
            logging.WARNING: '[%(levelname)s] %(name)s: %(message)s',
            logging.DEBUG: '[%(levelname)s] %(name)s: %(message)s'
        }

        verbose_formats = {
            logging.INFO: '%(asctime)s [%(levelname)s] %(name)s: %(message)s',
            logging.PROGRESS: '%(asctime)s [%(levelname)s] %(message)s',
            logging.RESULTS: '%(asctime)s [%(levelname)s] %(message)s',
            logging.WARNING: '%(asctime)s [%(levelname)s] %(name)s: %(message)s',
            logging.DEBUG: '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
        }

        self.non_verbose_formats = sorted(
            (level, logging.Formatter(fmt, **kwargs)) for level, fmt in non_verbose_formats.items()
        )

        self.verbose_formats = sorted(
            (level, logging.Formatter(fmt, **kwargs)) for level, fmt in verbose_formats.items()
        )

    def format(self, record):
        bisect_dict = self.non_verbose_formats
        if logging.root.level <= logging.DEBUG:
            bisect_dict = self.verbose_formats
        idx = bisect(bisect_dict, (record.levelno,), hi=len(bisect_dict)-1)
        _, formatter = bisect_dict[idx]
        return formatter.format(record)

class Logger(logging.getLoggerClass()):
    def progress(self, msg, *args, **kwargs):
        self.log(logging.PROGRESS, msg, *args, **kwargs)

    def results(self, msg, *args, **kwargs):
        self.log(logging.RESULTS, msg, *args, **kwargs)

logging.setLoggerClass(Logger)

def logger_basic_config(filename=None, level=None):
    if level is not None:
        level = level.upper() if isinstance(level, str) else int(level)
    logging.basicConfig(filename=filename, level=level)
    logging.root.handlers[0].formatter = LogFormatter()

@lru_cache
def get_terminal_cols():
    return shutil.get_terminal_size()[0]

def get_str_divider(div_char='-'):
    return str(div_char) * get_terminal_cols()

def get_divided_title(msg, div_char='-'):
    padding = max(2, len(msg) - get_terminal_cols())
    padding += padding % 2
    return "{pad}{msg}{pad}".format(pad=div_char * (padding // 2), msg=msg)

def print_divider(div_char='-'):
    print(get_str_divider(div_char=div_char))

def get_log_levels():
    return [x.lower() for x in logging._levelToName.values() if x.lower() != 'notset']

def get_cur_log_level(log):
    return logging._levelToName[log.getEffectiveLevel()].lower()

