##!/usr/bin/python3
"""
  Author:  H.Wang
  Purpose: Offer the most common methods dealing with a text file.
  Created: 5/2/2019
"""
import os
from functools import wraps


class TxtFileDealer:
    """

    """
    @staticmethod
    def readline(func):
        @wraps(func)
        def read_file_wraper(file, *args, **kwargs):
            for line in TxtFileDealer.yield_read_line(file):
                func(*args, **kwargs)
            return read_file_wraper

    @staticmethod
    def yield_read_line(file):
        with open(file) as f:
            for line in f:
                yield line
        f.close()