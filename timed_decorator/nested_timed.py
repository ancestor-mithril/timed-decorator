import gc
from functools import wraps
from gc import collect
from time import perf_counter_ns
from typing import Union

from .utils import nop, TimeFormatter, InputFormatter, synchronize_cuda, Logger, update_timing_dict

nested_level = 0
nested_times = dict()


def nested_timed(collect_gc: bool = True,
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
    A nested timing decorator that measures the time elapsed during the function call and accounts for other decorators
    further in the call stack.

    See Also:
        :class:`timed_decorator.simple_timed.timed` for parameter documentation.
    """
    gc_collect = collect if collect_gc else nop
    time_formatter = TimeFormatter(use_seconds, precision)
    input_formatter = InputFormatter(show_args, show_kwargs, display_level, sep)
    logger = Logger(stdout, file_path, logger_name)
    update_dict = update_timing_dict if out is not None else nop

    def decorator(fn):
        @wraps(fn)
        def wrap(*args, **kwargs):
            global nested_level, nested_times
            nested_level += 1

            gc_collect()
            if disable_gc:
                gc.disable()

            try:
                start = perf_counter_ns()
                ret = fn(*args, **kwargs)
                synchronize_cuda(*args, **kwargs)
                end = perf_counter_ns()
            except Exception as e:
                nested_level -= 1
                raise e
            finally:
                if disable_gc:
                    gc.enable()

            elapsed = end - start

            children_time = 0
            if nested_level in nested_times:
                children_time = sum(nested_times[nested_level])
                del nested_times[nested_level]

            own_time = elapsed - children_time
            nested_level -= 1

            if nested_level != 0:
                if nested_level not in nested_times:
                    nested_times[nested_level] = []
                nested_times[nested_level].append(elapsed)

            fn_name = fn.__qualname__ if use_qualname else fn.__name__
            update_dict(out, fn.__qualname__, elapsed, own_time)
            logger('\t' * nested_level + f'{input_formatter(fn_name, *args, **kwargs)} '
                                         f'-> total time: {time_formatter(elapsed)}, '
                                         f'own time: {time_formatter(own_time)}')
            if return_time:
                return ret, elapsed
            return ret

        return wrap

    return decorator
