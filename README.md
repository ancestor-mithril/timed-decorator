# timed-decorator

## Installation

Cloning the repo and installing with pip:

```
pip install git+https://github.com/ancestor-mithril/timed-decorator.git@master
```

## Usage

Attach it to the function you want to time and run the application. 

```py
import torch
from torch import Tensor

from timed.simple_timed import timed


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

Simple usage.
```py
from timed.simple_timed import timed


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
from timed.simple_timed import timed


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

The decorator can be configured to use seconds instead of nanoseconds. 
```py
from timed.simple_timed import timed


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
from timed.simple_timed import timed
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

The default display level is 1, and prints shapes where possible.

```py
from timed.simple_timed import timed
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

Displaying the keyword arguments.
```py
from timed.simple_timed import timed
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

Display level 2 prints the function arguments as given (not recommended).
```py
from timed.simple_timed import timed
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


