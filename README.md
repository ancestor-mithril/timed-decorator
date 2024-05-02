# timed-decorator

Simple and configurable timing decorator with little overhead that can be attached to python functions to measure their execution time.
Can easily display parameter types and lengths if available and is compatible with NumPy ndarrays, Pandas DataFrames and PyTorch tensors.


## Installation

```
pip install --upgrade timed-decorator
```

## Usage

Attach it to the function you want to time and run the application. 


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
# fibonacci() -> total time: 1114100ns
```

For more advanced usage, consider registering a timed decorator and using it afterward through your codebase. See [Registering a timed decorator](#registering-a-timed-decorator).

### Documentation

1. `timed`
    * `collect_gc` (`bool`): If `True`, runs a full garbage collection before timing the wrapped function. Default: `True`.
    * `disable_gc` (`bool`): If `True`, disabled garbage collection during function execution. Default: `False`.
    * `use_seconds` (`bool`): If `True`, displays the elapsed time in seconds. Default: `False`.
    * `precision` (`int`): Used in conjunction with `use_seconds`, represents the decimal points used for printing seconds. Default: `9`.
    * `show_args` (`bool`): If `True`, displays the function arguments according to `display_level`. Useful when timing function calls with arguments of different magnitude. Default: `False`.
    * `show_kwargs` (`bool`): If `True`, displays the keyword arguments according to `display_level`. Default: `False`.
    * `display_level` (`int`): The level of verbosity used when printing function arguments ad keyword arguments. If `0`, prints the type of the parameters. If `1`, prints values for all primitive types, shapes for arrays, tensors, dataframes and length for sequences. Otherwise, prints values for all parameters. Default: `1`.
    * `sep` (`str`): The separator used when printing function arguments and keyword arguments. Default: `', '`.
    * `stdout` (`bool`): If `True`, writes the elapsed time to stdout. Default: `True`.
    * `file_path` (`str`): If not `None`, writes the measurement at the end of the given file path. For thread safe file writing configure use `logger_name` instead. Default: `None`.
    * `logger_name` (`str`): If not `None`, uses the given logger to print the measurement. Can't be used in conjunction with `file_path`. Default: `None`. See [Using a logger](#using-a-logger).
    * `return_time` (`bool`): If `True`, returns the elapsed time in addition to the wrapped function's return value. Default: `False`.
    * `out` (`dict`): If not `None`, stores the elapsed time in nanoseconds in the given dict using the fully qualified function name as key, in the following format: (function call counts, total elapsed time, total "own time"). If the key already exists, updates the existing value. The elapsed time is equal to "own time" for the simple timed decorator. For the nested time decorator, the elapsed time is different from "own time" only when another function decorated with a nested timer is called during the execution of the current function. Default: `None`. See [Storing the elapsed time in a dict](#storing-the-elapsed-time-in-a-dict).
    * `use_qualname` (`bool`): If `True`, If `True`, uses the qualified name of the function when logging the elapsed time. Default: `False`.

2. `nested_timed` is similar to `timed`, however it is designed to work nicely with multiple timed functions that call each other, displaying both the total execution time and the difference after subtracting other timed functions on the same call stack. See [Nested timing decorator](#nested-timing-decorator).

3. `create_timed_decorator` registers a timed decorator with a given name.
   * `name` (`str`): The name of the timed decorator which will be instantiated using the provided arguments. Use this name for retrieving the timed decorator with `timed_decorator.builder.get_timed_decorator`.
   * `nested` (`bool`): If `True`, uses the `timed_decorator.nested_timed.nested_timed` as decorator, otherwise uses `timed_decorator.simple_timed.timed`. Default: `False`.
   * Also receives all the other arguments accepted by `timed` and `nested_timed`.

4. `get_timed_decorator` wraps the decorated function and lazily measures its elapsed time using the registered timed decorator. The timer can be registered after the function definition, but must be registered before the first function call.
   * `name` (`str`): The name of the timed decorator registered using `timed_decorator.builder.create_timed_decorator`.


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
# fibonacci() -> total time: 1114100ns
```

Getting both the function's return value and the elapsed time.
```py
from timed_decorator.simple_timed import timed


@timed(return_time=True)
def fibonacci(n: int) -> int:
    assert n > 0
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a


value, elapsed = fibonacci(10000)
print(f'10000th fibonacci number has {len(str(value))} digits. Calculating it took {elapsed}ns.')
# fibonacci() -> total time: 1001200ns
# 10000th fibonacci number has 2090 digits. Calculating it took 1001200ns.
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
# fibonacci() -> total time: 1062400ns
```

Using seconds instead of nanoseconds. 
```py
from timed_decorator.simple_timed import timed


@timed(disable_gc=True, use_seconds=True, precision=3)
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
# call_recursive_fibonacci() -> total time: 0.045s
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

Using the fully qualified name for timing. 

```py
from time import sleep
from timed_decorator.simple_timed import timed
class ClassA:
    @timed(use_qualname=True)
    def wait(self, x):
        sleep(x)
ClassA().wait(0.1)
# ClassA.wait() -> total time: 100943000ns
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


@timed(logger_name='TEST_LOGGER', stdout=False)
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


@timed(logger_name='TEST_LOGGER', stdout=False)
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

### Storing the elapsed time in a dict
```py
from time import sleep

from timed_decorator.simple_timed import timed

ns = {}


@timed(out=ns, stdout=False)
def fn():
    sleep(1)


fn()
print(ns)
fn()
print(ns)
```
Prints
```
{'fn': [1, 1000672000, 1000672000]}
{'fn': [2, 2001306900, 2001306900]}
```

### Compatible with PyTorch tensors

Synchronizes cuda device when cuda tensors are passed as function parameters.

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

if torch.cuda.is_available():
    a = a.cuda()
    b = b.cuda()
    batched_euclidean_distance(a, b)  # Cuda device is synchronized if function arguments are on device.
```
Prints:
```
batched_euclidean_distance(CpuTensor[10000, 800], CpuTensor[12000, 800]) -> total time: 685659400ns
batched_euclidean_distance(CudaTensor[10000, 800], CudaTensor[12000, 800]) -> total time: 260411900ns
```


### Registering a timed decorator

```py
from time import sleep

from timed_decorator.builder import create_timed_decorator, get_timed_decorator


@get_timed_decorator("MyCustomTimer")
def main():
    @get_timed_decorator("MyCustomTimer")
    def function_1():
        sleep(0.1)

    @get_timed_decorator("MyCustomTimer")
    def nested_function():
        @get_timed_decorator("MyCustomTimer")
        def function_2():
            sleep(0.2)

        @get_timed_decorator("MyCustomTimer")
        def function_3():
            sleep(0.3)

        function_2()
        function_2()
        function_3()

    nested_function()
    function_1()
    nested_function()
    function_1()


if __name__ == '__main__':
    my_measurements = {}
    create_timed_decorator("MyCustomTimer",
                           nested=False,  # This is true by default
                           collect_gc=False,  # I don't want to explicitly collect garbage
                           disable_gc=True,  # I don't want to wait for garbage collection during measuring
                           stdout=False,  # I don't wat to print stuff to console
                           out=my_measurements  # My measurements dict
                           )
    main()
    for key, (counts, elapsed, own_time) in my_measurements.items():
        print(f'Function {key} was called {counts} time(s) and took {elapsed / 1e+9}s')
    print()

    # Now I can do stuff with my measurements.
    functions = sorted(my_measurements.keys(), reverse=True)

    for i in range(len(functions)):
        fn_1 = functions[i]
        print(f'Function {fn_1}:')
        for j in range(i + 1, len(functions)):
            fn_2 = functions[j]
            if fn_1.startswith(fn_2):
                _, elapsed_1, _ = my_measurements[fn_1]
                _, elapsed_2, _ = my_measurements[fn_2]
                ratio = elapsed_1 / elapsed_2 * 100
                print(f'* took {ratio:.2f}% from {fn_2}')
        print()
```

Prints:
```
Function main.<locals>.nested_function.<locals>.function_2 was called 4 time(s) and took 0.8019482s
Function main.<locals>.nested_function.<locals>.function_3 was called 2 time(s) and took 0.6010157s
Function main.<locals>.nested_function was called 2 time(s) and took 1.403365s
Function main.<locals>.function_1 was called 2 time(s) and took 0.2007625s
Function main was called 1 time(s) and took 1.6043592s

Function main.<locals>.nested_function.<locals>.function_3:
* took 42.83% from main.<locals>.nested_function
* took 37.46% from main

Function main.<locals>.nested_function.<locals>.function_2:
* took 57.14% from main.<locals>.nested_function
* took 49.99% from main

Function main.<locals>.nested_function:
* took 87.47% from main

Function main.<locals>.function_1:
* took 12.51% from main

Function main:

```
