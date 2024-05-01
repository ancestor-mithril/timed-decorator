from functools import wraps
from typing import Union

from .nested_timed import nested_timed
from .simple_timed import timed

_timed_decorators = {}


def create_timed_decorator(name: str,
                           nested: bool = False,
                           collect_gc: bool = True,
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
    Registers a timed decorator with a given name. Once instantiated, the timed decorator can be retrieved with
    :class:`timed_decorator.builder.get_timed_decorator` and used for measuring the runtime of decorated functions.

    Args:
        name (str): The name of the timed decorator which will be instantiated using the provided arguments. Use this
            name for retrieving the timed decorator with :class:`timed_decorator.builder.get_timed_decorator`.
        nested (bool): If `True`, uses the :class:`timed_decorator.nested_timed.nested_timed` as decorator, otherwise
            uses :class:`timed_decorator.simple_timed.timed`. Default: `False`.

    See Also:
        :class:`timed_decorator.simple_timed.timed` for the remaining parameters' documentation.

    """
    global _timed_decorators
    if name in _timed_decorators:
        raise KeyError(f'Timed decorator {name} already registered.')

    decorator = nested_timed if nested else timed
    _timed_decorators[name] = decorator(collect_gc=collect_gc,
                                        disable_gc=disable_gc,
                                        use_seconds=use_seconds,
                                        precision=precision,
                                        show_args=show_args,
                                        show_kwargs=show_kwargs,
                                        display_level=display_level,
                                        sep=sep,
                                        stdout=stdout,
                                        file_path=file_path,
                                        logger_name=logger_name,
                                        return_time=return_time,
                                        out=out,
                                        use_qualname=use_qualname)


def _get_timed_decorator(name: str):
    global _timed_decorators
    if name not in _timed_decorators:
        raise KeyError(f'Timed decorator {name} not registered. Please register it first using '
                       f'timed_decorator.builder.create_timed_decorator')

    return _timed_decorators[name]


def get_timed_decorator(name: str):
    """
    Wraps the decorated function and lazily measures its elapsed time using the registered timed decorator. The timer
    can be registered after the function definition, but must be registered before the first function call.

    Args:
        name (str): The name of the timed decorator registered using
            :class:`timed_decorator.builder.create_timed_decorator`.

    """
    def decorator(fn):
        @wraps(fn)
        def wrap(*args, **kwargs):
            return _get_timed_decorator(name)(fn)(*args, **kwargs)

        return wrap

    return decorator
