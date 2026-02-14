from functools import wraps
from typing import Union

from .nested_timed import nested_timed
from .simple_timed import timed

_timed_decorators = {}


def create_timed_decorator(
    name: str,
    nested: bool = False,
    enabled: bool = True,
    collect_gc: bool = True,
    disable_gc: bool = False,
    use_seconds: bool = False,
    precision: int = 9,
    show_args: bool = False,
    show_kwargs: bool = False,
    display_level: int = 1,
    sep: str = ", ",
    stdout: bool = True,
    file_path: Union[str, None] = None,
    logger_name: Union[str, None] = None,
    return_time: bool = False,
    out: dict = None,
    use_qualname: bool = False,
):
    """
    Registers a timed decorator with a given name. Once instantiated, the timed decorator can be retrieved with
    :class:`timed_decorator.builder.get_timed_decorator` and used for measuring the runtime of decorated functions if
    enabled.

    Args:
        name (str): The name of the timed decorator which will be instantiated using the provided arguments. Use this
            name for retrieving the timed decorator with :class:`timed_decorator.builder.get_timed_decorator`.
        nested (bool): If `True`, uses the :class:`timed_decorator.nested_timed.nested_timed` as decorator, otherwise
            uses :class:`timed_decorator.simple_timed.timed`. Default: `False`.
        enabled (bool): If `True`, the timed decorator is enabled and used for timing decorated functions. Otherwise,
            functions decorated with `name` will not be timed. Default: `True`.

    See Also:
        :class:`timed_decorator.simple_timed.timed` for the remaining parameters' documentation.

    """
    if name in _timed_decorators:
        raise KeyError(f"Timed decorator {name} already registered.")

    decorator = nested_timed if nested else timed
    timer = decorator(
        collect_gc=collect_gc,
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
        use_qualname=use_qualname,
    )
    _timed_decorators[name] = (timer, enabled)


def _get_timed_decorator(name: str):
    if name not in _timed_decorators:
        raise KeyError(
            f"Timed decorator {name} not registered. Please register it first using "
            f"timed_decorator.builder.create_timed_decorator"
        )

    return _timed_decorators[name]


def get_timed_decorator(name: str):
    """
    Wraps the decorated function and lazily measures its elapsed time using the registered timed decorator. The timer
    can be registered after the function definition, but must be registered before the first function call. If the timer
    is disabled, the elapsed time will not be measured.

    Args:
        name (str): The name of the timed decorator registered using
            :class:`timed_decorator.builder.create_timed_decorator`.

    """

    def decorator(fn):
        @wraps(fn)
        def wrap(*args, **kwargs):
            timer, enabled = _get_timed_decorator(name)
            if enabled:
                return timer(fn)(*args, **kwargs)
            return fn(*args, **kwargs)

        return wrap

    return decorator


def apply_timed_decorator(fn, name: str = None):
    """
    Applies a timed decorator to an arbitrary function. This can be used to apply timing decorators to functions
    from external libraries or any codebase without changing the source code.

    By default, applies the :class:`timed_decorator.simple_timed.timed` decorator with default parameters. If a
    ``name`` is provided, uses the registered timed decorator with that name (see
    :class:`timed_decorator.builder.create_timed_decorator`). The named decorator is lazily resolved, meaning it
    can be registered after applying the decorator but before the first function call.

    Args:
        fn: The function to be decorated.
        name (str): The name of a registered timed decorator. If ``None``, applies the default
            :class:`timed_decorator.simple_timed.timed` decorator. Default: ``None``.

    Returns:
        The decorated function.

    """
    if name is not None:
        @wraps(fn)
        def wrap(*args, **kwargs):
            timer, enabled = _get_timed_decorator(name)
            if enabled:
                return timer(fn)(*args, **kwargs)
            return fn(*args, **kwargs)

        return wrap

    return timed()(fn)
