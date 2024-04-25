from functools import wraps
from gc import collect
from time import perf_counter_ns

from timed.utils import nop, TimeFormatter, InputFormatter, synchronize_cuda


def timed(collect_gc: bool = True,
          use_seconds: bool = False,
          precision: int = 9,
          show_args: bool = False,
          show_kwargs: bool = False,
          display_level: int = 1,
          sep: str = ', '):
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
            total_time = end - start

            print(f'{input_formatter(fn.__name__, *args, **kwargs)} -> total time: {time_formatter(total_time)}')
            return ret

        return wrap

    return decorator
