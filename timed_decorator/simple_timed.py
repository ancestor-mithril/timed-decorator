import gc
from functools import wraps
from gc import collect
from time import perf_counter_ns
from typing import Union

from .utils import nop, TimeFormatter, InputFormatter, synchronize_cuda, Logger, update_timing_dict


def timed(collect_gc: bool = True,
          disable_gc: bool = False,
          use_seconds: bool = False,
          precision: int = 9,
          show_args: bool = False,
          show_kwargs: bool = False,
          display_level: int = 1,
          sep: str = ', ',
          stdout: bool = True,
          file_path: Union[str, None] = None,
          logger_name: Union[str, None] = None,
          return_time: bool = False,
          out: dict = None,
          use_qualname: bool = False):
    """
    A simple timing decorator that measures the time elapsed during the function call and prints it.
    It uses perf_counter_ns for measuring which includes time elapsed during sleep and is system-wide.

    Args:
        collect_gc (bool): If `True`, runs a full garbage collection before timing the wrapped function. Default:
            `True`.
        disable_gc (bool): If `True`, disabled garbage collection during function execution. Default: `False`.
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
        stdout (bool): If `True`, writes the elapsed time to stdout. Default: `True`.
        file_path (str): If not `None`, writes the measurement at the end of the given file path. For thread safe
            file writing configure use `logger_name` instead. Default: `None`.
        logger_name (str): If not `None`, uses the given logger to print the measurement. Can't be used in conjunction
            with `file_path`. Default: `None`.
        return_time (bool): If `True`, returns the elapsed time in addition to the wrapped function's return value.
            Default: `False`.
        out (dict): If not `None`, stores the elapsed time in nanoseconds in the given dict using the fully qualified
            function name as key, in the following format: (function call counts, total elapsed time, total "own time").
            If the key already exists, updates the existing value. The elapsed time is equal to "own time" for the simple
            timed decorator. For the nested time decorator, the elapsed time is different from "own time" only when
            another function decorated with a nested timer is called during the execution of the current function.
            Default: `None`.
        use_qualname (bool): If `True`, uses the qualified name of the function when logging the elapsed time. Default:
            `False`.
    """
    gc_collect = collect if collect_gc else nop
    time_formatter = TimeFormatter(use_seconds, precision)
    input_formatter = InputFormatter(show_args, show_kwargs, display_level, sep)
    logger = Logger(stdout, file_path, logger_name)
    update_dict = update_timing_dict if out is not None else nop

    def decorator(fn):
        @wraps(fn)
        def wrap(*args, **kwargs):
            gc_collect()
            if disable_gc:
                gc.disable()

            try:
                start = perf_counter_ns()
                ret = fn(*args, **kwargs)
                synchronize_cuda(*args, **kwargs)
                end = perf_counter_ns()
            finally:
                if disable_gc:
                    gc.enable()

            elapsed = end - start
            fn_name = fn.__qualname__ if use_qualname else fn.__name__
            update_dict(out, fn.__qualname__, elapsed, elapsed)
            logger(f'{input_formatter(fn_name, *args, **kwargs)} -> total time: {time_formatter(elapsed)}')
            if return_time:
                return ret, elapsed
            return ret

        return wrap

    return decorator
