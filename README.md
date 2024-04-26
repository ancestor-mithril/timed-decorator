# timed-decorator

## Installation

```
pip install timed-decorator
```

## Usage

Attach it to the function you want to time and run the application. 

```py
import torch
from torch import Tensor

from timed_decorator.simple_timed import timed


@timed(show_args=True)
def batched_euclidean_distance(x: Tensor, y: Tensor) -> Tensor:
    diff = x @ y.T
    x_squared = (x ** 2).sum(dim=1)
    y_squared = (b ** 2).sum(dim=1)
    return x_squared.unsqueeze(-1) + y_squared.unsqueeze(0) - 2 * diff


a = torch.rand((10000, 800))
b = torch.rand((12000, 800))
batched_euclidean_distance(a, b)
a = a.cuda()
b = b.cuda()
batched_euclidean_distance(a, b)
```
Prints:
```
batched_euclidean_distance(CpuTensor[10000, 800], CpuTensor[12000, 800]) -> total time: 685659400ns
batched_euclidean_distance(CudaTensor[10000, 800], CudaTensor[12000, 800]) -> total time: 260411900ns
```

### Documentation

1. `timed`
    * `collect_gc` (`bool`): If `True`, runs a full garbage collection before timing the wrapped function. Default: `True`.
    * `use_seconds` (`bool`): If `True`, displays the elapsed time in seconds. Default: `False`.
    * `precision` (`int`): Used in conjunction with `use_seconds`, represents the decimal points used for printing seconds. Default: `9`.
    * `show_args` (`bool`): If `True`, displays the function arguments according to `display_level`. Useful when timing function calls with arguments of different magnitude. Default: `False`.
    * `show_kwargs` (`bool`): If `True`, displays the keyword arguments according to `display_level`. Default: `False`.
    * `display_level` (`int`): The level of verbosity used when printing function arguments ad keyword arguments. If `0`, prints the type of the parameters. If `1`, prints values for all primitive types, shapes for arrays, tensors, dataframes and length for sequences. Otherwise, prints values for all parameters. Default: `1`.
    * `sep` (`str`): The separator used when printing function arguments and keyword arguments. Default: `', '`.
    * `file_path` (str): If not `None`, writes the measurement at the end of the given file path. For thread safe file writing configure use `logger_name` instead. Can't be used in conjunction with `logger_name`. If both `file_path` and `logger_name` are `None`, writes to stdout. Default: `None`.
    * `logger_name` (str): If not `None`, uses the given logger to print the measurement. Can't be used in conjunction with `file_path`. If both `file_path` and `logger_name` are `None`, writes to stdout. Default: `None`. See [Using a logger](#using-a-logger).

2. `nested_timed` is similar to `timed`, however it is designed to work nicely with multiple timed functions that call each other, displaying both the total execution time and the difference after substracting other timed functions on the same call stack. See [Nested timing decorator](#nested-timing-decorator).

### Examples

Simple usage.
```py
from timed_decorator.simple_timed import timed


@timed()
def fibonacci(n: int) -> int:
    assert n > 0
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a


fibonacci(10000)
# fibonacci() -> total time: 2114100ns
```

Set `collect_gc=False` to disable pre-collection of garbage.

```py
from timed_decorator.simple_timed import timed


@timed(collect_gc=False)
def fibonacci(n: int) -> int:
    assert n > 0
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a


fibonacci(10000)
# fibonacci() -> total time: 2062400ns
```

Using seconds instead of nanoseconds. 
```py
from timed_decorator.simple_timed import timed


@timed(use_seconds=True, precision=3)
def call_recursive_fibonacci(n: int) -> int:
    return recursive_fibonacci(n)


def recursive_fibonacci(n: int) -> int:
    assert n > 0
    if n > 3:
        return recursive_fibonacci(n - 1) + recursive_fibonacci(n - 2)
    if n == 1:
        return 0
    return 1


call_recursive_fibonacci(30)
# call_recursive_fibonacci() -> total time: 0.098s
```

Displaying function parameters:
```py
from timed_decorator.simple_timed import timed
import numpy as np


@timed(show_args=True, display_level=0)
def numpy_operation(array_list, single_array, inplace=False, aggregate='mean', weights=None):
    x = np.array(array_list)
    if weights is not None:
        x = (x.T * weights).T

    if aggregate == 'mean':
        x = x.mean(axis=0)
    else:
        x = x.sum(axis=0)

    if inplace:
        single_array += x
        return single_array
    else:
        other_array = single_array + x
        return other_array


numpy_operation(
    [np.random.rand(2, 3) for _ in range(10)],
    np.random.rand(2, 3),
    weights=[i / 10 for i in range(10)],
    inplace=True
)
# numpy_operation(list, ndarray) -> total time: 204200ns
```

Using the default display level (1).

```py
from timed_decorator.simple_timed import timed
import numpy as np


@timed(show_args=True)
def numpy_operation(array_list, single_array, inplace=False, aggregate='mean', weights=None):
    x = np.array(array_list)
    if weights is not None:
        x = (x.T * weights).T

    if aggregate == 'mean':
        x = x.mean(axis=0)
    else:
        x = x.sum(axis=0)

    if inplace:
        single_array += x
        return single_array
    else:
        other_array = single_array + x
        return other_array


numpy_operation(
    [np.random.rand(2, 3) for _ in range(10)],
    np.random.rand(2, 3),
    weights=[i / 10 for i in range(10)],
    inplace=True,
    aggregate='sum'
)
# numpy_operation(list(ndarray)[10], ndarray(2, 3)) -> total time: 166400ns
```

Showing the keyword arguments.

```py
from timed_decorator.simple_timed import timed
import numpy as np


@timed(show_args=True, show_kwargs=True)
def numpy_operation(array_list, single_array, inplace=False, aggregate='mean', weights=None):
    x = np.array(array_list)
    if weights is not None:
        x = (x.T * weights).T

    if aggregate == 'mean':
        x = x.mean(axis=0)
    else:
        x = x.sum(axis=0)

    if inplace:
        single_array += x
        return single_array
    else:
        other_array = single_array + x
        return other_array


numpy_operation(
    [np.random.rand(2, 3) for _ in range(10)],
    np.random.rand(2, 3),
    weights=[i / 10 for i in range(10)],
    inplace=True,
    aggregate='sum'
)
# numpy_operation(list(ndarray)[10], ndarray(2, 3), ('weights', 'list(float)[10]'), ('inplace', 'True'), ('aggregate', 'sum')) -> total time: 166400ns
```

Not recommended: using display level 2 shows unformatted function arguments.

```py
from timed_decorator.simple_timed import timed
import numpy as np


@timed(show_args=True, show_kwargs=True, display_level=2)
def numpy_operation(array_list, single_array, inplace=False, aggregate='mean', weights=None):
    x = np.array(array_list)
    if weights is not None:
        x = (x.T * weights).T

    if aggregate == 'mean':
        x = x.mean(axis=0)
    else:
        x = x.sum(axis=0)

    if inplace:
        single_array += x
        return single_array
    else:
        other_array = single_array + x
        return other_array


numpy_operation(
    [np.random.rand(1, 3) for _ in range(1)],
    np.random.rand(1, 3),
    weights=[i / 10 for i in range(1)],
    inplace=True
)
# numpy_operation([array([[0.74500602, 0.70666224, 0.83888559]])], [[0.74579988 0.51878032 0.06419635]], ('weights', '[0.0]'), ('inplace', 'True')) -> total time: 185300ns
```

### Nested timing decorator

```py
from time import sleep

from timed_decorator.nested_timed import nested_timed


@nested_timed()
def nested_fn():
    @nested_timed()
    def sleeping_fn(x):
        sleep(x)

    @nested_timed()
    def other_fn():
        sleep(0.5)
        sleeping_fn(0.5)

    sleep(1)
    sleeping_fn(1)
    other_fn()
    sleeping_fn(1)


nested_fn()
```
Prints
```
        sleeping_fn() -> total time: 1000592700ns, own time: 1000592700ns
                sleeping_fn() -> total time: 500687200ns, own time: 500687200ns
        other_fn() -> total time: 1036725800ns, own time: 536038600ns
        sleeping_fn() -> total time: 1000705600ns, own time: 1000705600ns
nested_fn() -> total time: 4152634300ns, own time: 1114610200ns
```

### Using a logger
```py
import logging
from time import sleep

from timed_decorator.simple_timed import timed

logging.basicConfig()
logging.root.setLevel(logging.NOTSET)


@timed(logger_name='TEST_LOGGER')
def fn():
    sleep(1)


fn()
fn()
```
Prints
```
INFO:TEST_LOGGER:fn() -> total time: 1000368900ns
INFO:TEST_LOGGER:fn() -> total time: 1001000200ns
```

Capture a logger's input
```py
import logging
from io import StringIO
from time import sleep

from timed_decorator.simple_timed import timed

log_stream = StringIO()
log_handler = logging.StreamHandler(log_stream)
logging.root.setLevel(logging.NOTSET)
logging.getLogger('TEST_LOGGER').addHandler(log_handler)


@timed(logger_name='TEST_LOGGER')
def fn():
    sleep(1)


fn()
fn()

print(log_stream.getvalue().split('\n')[:-1])
```
Prints
```
['fn() -> total time: 1000214700ns', 'fn() -> total time: 1000157800ns']
```
