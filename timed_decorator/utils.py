import logging
from collections.abc import Sequence
from typing import Union

try:
    from numpy import ndarray
except ModuleNotFoundError:
    class ndarray:
        pass
try:
    from pandas import DataFrame
    from pandas import Series
except ModuleNotFoundError:
    class DataFrame:
        pass


    class Series:
        pass
try:
    import torch
    from torch import Tensor
except ModuleNotFoundError:
    class Tensor:
        pass


    torch = None


def nop(*args, **kwargs):
    pass


def build_decorated_fn(fn, decorator, **decorator_kwargs):
    @decorator(**decorator_kwargs)
    def decorated(*args, **kwargs):
        return fn(*args, **kwargs)

    return decorated


class TimeFormatter:
    def __init__(self, use_seconds: bool = False, precision: int = 9):
        self.use_seconds = use_seconds
        self.precision = precision

    def __call__(self, nanoseconds):
        if self.use_seconds:
            return f'{nanoseconds / 1e9:.{self.precision}f}s'
        return f'{nanoseconds}ns'


class Logger:
    def __init__(self, file_path: Union[str, None], logger_name: Union[str, None]):
        assert file_path is None or logger_name is None

        self.file_path = file_path
        self.logger_name = logger_name

    def __call__(self, string: str):
        if self.file_path is not None:
            with open(self.file_path, 'a') as f:
                f.write(string + '\n')
        elif self.logger_name is not None:
            logging.getLogger(self.logger_name).info(string)
        else:
            print(string)


class InputFormatter:
    def __init__(self, show_args: bool = False, show_kwargs: bool = False, display_level: int = 1, sep: str = ', '):
        self.show_args = show_args
        self.show_kwargs = show_kwargs
        self.display_level = display_level
        self.sep = sep

    def __call__(self, fn_name, *args, **kwargs):
        parameters = ''
        if self.show_args:
            parameters = self.format_args(*args)
        if self.show_kwargs:
            kwargs = self.format_kwargs(**kwargs)
            if len(parameters) > 0:
                parameters += self.sep + kwargs
            else:
                parameters = kwargs

        return f'{fn_name}({parameters})'

    def format(self, x) -> str:
        if self.display_level == 0:
            return type(x).__name__

        if self.display_level == 1:
            if isinstance(x, (ndarray, DataFrame, Series)):
                return f'{type(x).__name__}{x.shape}'

            if isinstance(x, Tensor):
                return f'{x.device.type.capitalize()}{type(x).__name__}{str(x.shape).lstrip("torch.Size(").rstrip(")")}'

            if isinstance(x, (str, int, float, bool)):
                return str(x)

            if isinstance(x, Sequence) and len(x) > 0:
                return f'{type(x).__name__}({type(x[0]).__name__})[{len(x)}]'

            if hasattr(x, '__len__'):
                return f'{type(x).__name__}[{len(x)}]'

            return type(x).__name__

        return str(x)

    def format_args(self, *args) -> str:
        return ', '.join([self.format(i) for i in args])

    def format_kwargs(self, **kwargs):
        ret = [(k, self.format(v)) for k, v in kwargs.items()]
        ret = map(str, ret)
        return self.sep.join(ret)


def get_tensor_device(args):
    for i in args:
        if isinstance(i, Tensor) and i.device.type == 'cuda':
            return i.device
        if isinstance(i, Sequence) and len(i) > 0 and isinstance(i[0], Tensor) and i[0].device.type == 'cuda':
            return i[0].device
    return ''


def synchronize_cuda(*args, **kwargs):
    if torch is None:
        return
    device = get_tensor_device(args)
    if device == '':
        device = get_tensor_device(kwargs.values())
    if device != '':
        torch.cuda.synchronize(device)
