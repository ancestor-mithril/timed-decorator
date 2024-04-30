from typing import Union


def create_timed_decorator(name: str,
                           nested: bool = True,
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
    Instantiates the timed decorator with a given name. Once instantiated, the timed decorator can be retrieved with
    :class:`timed_decorator.builder.get_timed_decorator` and used for measuring the runtime of decorated functions.

    Args:
        name (str): The name of the timed decorator which will be instantiated using the provided arguments. Use this
            name for retrieving the timed decorator with :class:`timed_decorator.builder.get_timed_decorator`.
        nested (bool): If `True`, uses the :class:`timed_decorator.nested_timed.nested_timed` as decorator, otherwise
            uses :class:`timed_decorator.simple_timed.timed`.

    See Also:
        :class:`timed_decorator.simple_timed.timed` for the remaining parameters' documentation.
    """
    pass
