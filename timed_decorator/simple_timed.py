from functools import wraps
from gc import collect
from time import perf_counter_ns
from typing import Union

from .utils import nop, TimeFormatter, InputFormatter, synchronize_cuda, Logger


def timed(collect_gc: bool = True,
          use_seconds: bool = False,
          precision: int = 9,
          show_args: bool = False,
          show_kwargs: bool = False,
          display_level: int = 1,
          sep: str = ', ',
          file_path: Union[str, None] = None,
          logger_name: Union[str, None] = None):
    """
    A simple timing decorator that measures the time elapsed during the function call and prints it.
    It uses perf_counter_ns for measuring which includes time elapsed during sleep and is system-wide.

    Args:
        collect_gc (bool): If `True`, runs a full garbage collection before timing the wrapped function. Default:
            `True`.
        use_seconds (bool): If `True`, displays the elapsed time in seconds. Default: `False`.
        precision (int): Used in conjunction with `use_seconds`, represents the decimal points used for printing
            seconds. Default: `9`.
        show_args (bool): If `True`, displays the function arguments according to `display_level`. Useful when timing
            function calls with arguments of different magnitude. Default: `False`.
        show_kwargs (bool): If `True`, displays the keyword arguments according to `display_level`. Default: `False`.
        display_level (int): The level of verbosity used when printing function arguments ad keyword arguments. If `0`,
            prints the type of the parameters. If `1`, prints values for all primitive types, shapes for arrays,
            tensors, dataframes and length for sequences. Otherwise, prints values for all parameters. Default: `1`.
        sep (str): The separator used when printing function arguments and keyword arguments. Default: `', '`.
        file_path (str): If not `None`, writes the measurement at the end of the given file path. For thread safe
            file writing configure use `logger_name` instead. Can't be used in conjunction with `logger_name`. If both
            `file_path` and `logger_name` are `None`, writes to stdout. Default: `None`.
        logger_name (str): If not `None`, uses the given logger to print the measurement. Can't be used in conjunction
            with `file_path`. If both `file_path` and `logger_name` are `None`, writes to stdout. Default: `None`.
    """
    assert file_path is None or logger_name is None

    gc_collect = collect if collect_gc else nop
    time_formatter = TimeFormatter(use_seconds, precision)
    input_formatter = InputFormatter(show_args, show_kwargs, display_level, sep)
    logger = Logger(file_path, logger_name)

    def decorator(fn):
        @wraps(fn)
        def wrap(*args, **kwargs):
            gc_collect()

            start = perf_counter_ns()
            ret = fn(*args, **kwargs)
            synchronize_cuda(*args, **kwargs)
            end = perf_counter_ns()

            logger(f'{input_formatter(fn.__name__, *args, **kwargs)} -> total time: {time_formatter(end - start)}')
            return ret

        return wrap

    return decorator
