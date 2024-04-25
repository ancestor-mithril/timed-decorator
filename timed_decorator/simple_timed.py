from functools import wraps
from gc import collect
from time import perf_counter_ns

from .utils import nop, TimeFormatter, InputFormatter, synchronize_cuda


def timed(collect_gc: bool = True,
          use_seconds: bool = False,
          precision: int = 9,
          show_args: bool = False,
          show_kwargs: bool = False,
          display_level: int = 1,
          sep: str = ', '):
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
    """
    gc_collect = collect if collect_gc else nop
    time_formatter = TimeFormatter(use_seconds, precision)
    input_formatter = InputFormatter(show_args, show_kwargs, display_level, sep)

    def decorator(fn):
        @wraps(fn)
        def wrap(*args, **kwargs):
            gc_collect()

            start = perf_counter_ns()
            ret = fn(*args, **kwargs)
            synchronize_cuda(*args, **kwargs)
            end = perf_counter_ns()

            print(f'{input_formatter(fn.__name__, *args, **kwargs)} -> total time: {time_formatter(end - start)}')
            return ret

        return wrap

    return decorator
